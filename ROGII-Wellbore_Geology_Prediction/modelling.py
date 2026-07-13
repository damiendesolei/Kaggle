# -*- coding: utf-8 -*-
"""
Spyder Editor

https://www.kaggle.com/code/damiendesolei/rogii-wellbore-geology-lightgbm-baseline?scriptVersionId=327105716
"""

from pathlib import Path
from collections import defaultdict
import gc, warnings, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)



SEED = 42
RNG = np.random.default_rng(SEED)


TUNE = True
N_FOLDS = 5
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=0,
    seed=SEED,
    device_type='gpu'
)
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100



DATA_ROOT = Path(r"H:\kaggle\rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
print("DATA_ROOT:", DATA_ROOT)
print("Train dir contents (head):", sorted(p.name for p in TRAIN_DIR.iterdir())[:5])
print("Test  dir contents (head):", sorted(p.name for p in TEST_DIR.iterdir())[:5])


def well_id_from(path: Path) -> str:
    return path.name.split("__")[0].replace(".png", "")

def list_wells(split_dir: Path):
    return [well_id_from(p) for p in sorted(split_dir.glob("*__horizontal_well.csv"))]

def horiz_path(wid, split):
    return (TRAIN_DIR if split == "train" else TEST_DIR) / f"{wid}__horizontal_well.csv"

def type_path(wid, split):
    return (TRAIN_DIR if split == "train" else TEST_DIR) / f"{wid}__typewell.csv"

train_wells = list_wells(TRAIN_DIR)
test_wells  = list_wells(TEST_DIR)
print(f"Training wells: {len(train_wells)}")
print(f"Test wells:     {len(test_wells)}")


def robust_slope(x, y, default=0.0):
    """Slope of y on x via least squares; returns default on degenerate input."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2: return default
    xx = x[m] - x[m].mean()
    yy = y[m] - y[m].mean()
    denom = float(np.sum(xx * xx))
    if denom <= 1e-12: return default
    return float(np.sum(xx * yy) / denom)

def first_nan_idx(s: pd.Series):
    mask = s.isna()
    if not mask.any(): return None
    return int(mask.idxmax()) if mask.iloc[0] else int(np.argmax(mask.values))



#CZ 20260709
DENSE_SPW = 60   # ANCC sample points taken per training well when building the index
DENSE_K   = 20   # neighbours used per query point

class DenseANCCImputer:
    """Spatial KNN imputer for the ANCC formation column.

    Built once from the training wells only (test wells don't carry ANCC).
    For any (X, Y) query point, `impute` inverse-distance-weights the k
    nearest sampled ANCC points from OTHER wells and returns:
      - ancc_pred:  the imputed ANCC value at that location
      - ancc_std:   the (weighted) spread across those k neighbours
      - dense_dist: distance to the nearest of the k neighbours used — a
                    proxy for how trustworthy the imputation is (small =
                    nearby wells with known ANCC exist; large = extrapolating
                    into a spatially sparse region).
    """
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))

        if not xs:
            raise ValueError("No wells with usable X/Y/ANCC data were found "
                              "while building DenseANCCImputer.")

        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        # Per-axis scaling so X and Y contribute comparably to distance.
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_query, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_query = np.atleast_2d(xy_query)
        q = xy_query / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)

        # Exclude the query well's own sampled points (avoids leakage on train).
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)

        order = np.argpartition(dist, min(k - 1, nf - 1), axis=1)[:, :k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)

        valid = np.isfinite(dk)
        w = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
        w_sum = w.sum(1)
        safe = np.where(w_sum < 1e-9, 1.0, w_sum)

        neighbour_ancc = self.ancc[ik]
        ancc_pred = (neighbour_ancc * w).sum(1) / safe
        ancc_pred = np.where(w_sum < 1e-9, float(self.ancc.mean()), ancc_pred)

        var = ((neighbour_ancc - ancc_pred[:, None]) ** 2 * w).sum(1) / safe
        ancc_std = np.sqrt(np.maximum(var, 0.0))

        dense_dist = np.where(valid, dk, np.inf).min(1)

        return (ancc_pred.astype(np.float32),
                ancc_std.astype(np.float32),
                dense_dist.astype(np.float32))



# CZ 2060711 spatial_knn_dist
PLANE_K = 10       # neighbouring wells used per query
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

class FormationPlaneKNN:
    """Per-well weighted-least-squares plane fit over the formation columns.

    Unlike DenseANCCImputer (many sample points per well), this collapses
    each training well to ONE representative point — median X, median Y, and
    median value of each formation column — then, for a query (X, Y), fits a
    weighted-least-squares plane z = a*X + b*Y + c through the k nearest
    neighbour wells (inverse-distance weighted) for every formation column
    simultaneously. `impute` returns:
      - form_pred: predicted value of each formation column at the query point
      - spatial_knn_dist: distance to the nearest neighbour well actually
        used — how spatially isolated this well is from others with known
        formation picks (large => extrapolating into sparse territory).
    """
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)

        if not rows:
            raise ValueError("No wells with usable X/Y/formation data found "
                              "while building FormationPlaneKNN.")

        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_query, self_wid=None, k=PLANE_K):
        xy_query = np.atleast_2d(xy_query)
        q = xy_query / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)

        # Exclude the query well itself (avoids leakage on train).
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)

        order = np.argpartition(dist, min(k - 1, nf - 1), axis=1)[:, :k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)

        valid = np.isfinite(dk)
        w = np.where(valid, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]
        wx = w * xn; wy = w * yn

        n = len(q)
        A = np.zeros((n, 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1];       A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2];       A[:, 2, 1] = A[:, 1, 2];       A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9

        rhs = np.stack([(wx[:, :, None] * fn).sum(1),
                         (wy[:, :, None] * fn).sum(1),
                         (w[:, :, None] * fn).sum(1)], axis=1)
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((n, 3, len(FORMATIONS)))
            for r in range(n):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except np.linalg.LinAlgError:
                    pass

        Xq = xy_query[:, 0]; Yq = xy_query[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~valid.any(1)] = self.fa.mean(0)

        spatial_knn_dist = np.where(valid, dk, np.inf).min(1).astype(np.float32)
        return pred, spatial_knn_dist
    
    
# CZ 20260711 particle filter
try:
    from numba import njit
except ImportError as e:
    raise ImportError(
        "pf_ancc_delta needs the 'numba' package for the particle-filter kernel "
        "to run fast enough — install it with `pip install numba`."
    ) from e

PF_ANCC_N = 600
PF_ANCC_ALPHA = 0.998        # momentum on the particle's rate of change
PF_ANCC_RN = 0.002           # process noise on the rate
PF_ANCC_PN = 0.005           # process noise on the position
PF_ANCC_INIT_SPREAD = 0.3    # initial particle spread around the anchor
PF_ANCC_RP = 0.1             # resample jitter on position
PF_ANCC_RR = 0.001           # resample jitter on rate
PF_ANCC_RESAMP = 0.5         # resample when effective sample size < RESAMP * N
PF_ANCC_GR_SIG_MIN, PF_ANCC_GR_SIG_MAX, PF_ANCC_GR_SIG_DEF = 10.0, 60.0, 30.0


def _typewell_grid(tw_tvt, tw_gr, step=0.2):
    """Dense, regularly-spaced GR lookup table over the typewell's TVT range,
    so the particle filter can interpolate 'expected GR at this TVT' in O(1)
    inside the hot inner loop instead of doing a searchsorted every particle."""
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    grid_tvt = np.arange(tmin, tmax + step, step)
    return np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


@njit(cache=True)
def _pf_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _pf_resample(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    new_pos = np.empty(N); new_aux = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        new_pos[j] = pos[ci] + rp * np.random.randn()
        new_aux[j] = aux[ci] + rv * np.random.randn()
    return new_pos, new_aux


@njit(cache=True)
def _pf_ancc_kernel(md_v, z_v, gr_v, grid, vmin, step, gr_sigma, start_pos, init_rate, N,
                     ALPHA, RN, PN, INIT_SPREAD, RP, RR, RESAMP):
    """Particle filter over `position = TVT + Z` (so a particle's implied TVT is
    position - Z at that row). Particles drift forward using a momentum-smoothed
    rate; each GR reading re-weights particles by how well their implied TVT's
    typewell GR matches the well's actual GR. Returns the weighted-mean TVT and
    its weighted std at every row."""
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = start_pos + INIT_SPREAD * np.random.randn()
        rate[j] = init_rate + 0.01 * np.random.randn()

    n = len(md_v)
    pts = np.empty(n); stds = np.empty(n)
    prev_md = md_v[0] - 1.0

    for i in range(n):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            lo = vmin - 50.0
            hi = vmin + len(grid) * step + 50.0
            if tvt_j < lo: tvt_j = lo
            if tvt_j > hi: tvt_j = hi
            pos[j] = tvt_j + z_v[i]

        if not np.isnan(gr_v[i]):
            wsum = 0.0
            for j in range(N):
                expected_gr = _interp1(grid, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - expected_gr) / gr_sigma
                dd = d * d
                lk = np.exp(-0.5 * dd) if dd < 600.0 else 0.0
                if lk < 1e-300: lk = 1e-300
                w[j] *= lk
                wsum += w[j]
            if wsum > 0.0:
                for j in range(N): w[j] /= wsum
            else:
                for j in range(N): w[j] = 1.0 / N

        neff = 0.0
        for j in range(N): neff += w[j] * w[j]
        neff = 1.0 / neff
        if neff < RESAMP * N:
            pos, rate = _pf_resample(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1.0 / N

        tvt_est = 0.0
        for j in range(N): tvt_est += w[j] * (pos[j] - z_v[i])
        pts[i] = tvt_est
        var = 0.0
        for j in range(N): var += w[j] * (pos[j] - z_v[i] - tvt_est) ** 2
        stds[i] = var ** 0.5
        prev_md = md_v[i]

    return pts, stds


def run_pf_ancc(h, eval_row_idx, tw_clean, last_TVT, last_MD, last_Z, visible, N=PF_ANCC_N, seed=SEED):
    """Run the ANCC-anchored particle filter over ALL hidden rows for this well
    (eval_row_idx, in MD order — the full post-anchor sequence, not just the
    labeled subset), returning (pf_tvt, pf_std) aligned to that index. Falls
    back to a flat last_TVT prediction if there's no usable typewell to
    compare GR against."""
    n = len(eval_row_idx)
    if n == 0:
        return np.array([], np.float32), np.array([], np.float32)
    if tw_clean is None or len(tw_clean) < 3:
        return np.full(n, last_TVT, np.float32), np.zeros(n, np.float32)

    tw_tvt = tw_clean["TVT"].to_numpy(np.float64)
    tw_gr  = tw_clean["GR"].to_numpy(np.float64)
    grid, vmin, step = _typewell_grid(tw_tvt, tw_gr)

    # GR noise level: spread of known-zone GR around the typewell's expected GR.
    kn_gr = visible["GR"].values if "GR" in visible.columns else np.array([])
    if len(kn_gr):
        expected_at_kn = np.interp(visible["TVT_input"].values, tw_tvt, tw_gr)
        gr_sigma = float(np.clip(np.nanstd(kn_gr - expected_at_kn),
                                  PF_ANCC_GR_SIG_MIN, PF_ANCC_GR_SIG_MAX))
    else:
        gr_sigma = PF_ANCC_GR_SIG_DEF

    # Initial rate of change of (TVT + Z) vs MD, estimated from the last 30 known rows.
    tail = visible.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dmd = np.diff(tail["MD"].values)
    m = dmd > 0
    init_rate = float(np.median((dt + dz)[m] / dmd[m])) if m.sum() >= 3 else 0.0

    md_v = h["MD"].values[eval_row_idx].astype(np.float64)
    z_v  = h["Z"].values[eval_row_idx].astype(np.float64)
    gr_series = pd.Series(h["GR"].values).interpolate(limit_direction="both")
    gr_v = gr_series.values[eval_row_idx].astype(np.float64)

    start_pos = last_TVT + last_Z
    if seed is not None:
        _pf_seed(int(seed))
    pts, stds = _pf_ancc_kernel(md_v, z_v, gr_v, grid, vmin, step, gr_sigma, start_pos, init_rate, N,
                                 PF_ANCC_ALPHA, PF_ANCC_RN, PF_ANCC_PN, PF_ANCC_INIT_SPREAD,
                                 PF_ANCC_RP, PF_ANCC_RR, PF_ANCC_RESAMP)
    return pts.astype(np.float32), stds.astype(np.float32)


# ---- second, independent tracker: TVT directly, velocity-constrained against Z ----
PF_Z_N = 600
PF_Z_MOM = 0.993          # momentum on TVT velocity
PF_Z_VN = 0.005           # process noise on velocity
PF_Z_PN = 0.01            # process noise on position (TVT)
PF_Z_GR_WT = 0.3          # weight of smoothed-GR likelihood vs raw-GR likelihood
PF_Z_ROUGH_P = 0.2        # resample jitter on position
PF_Z_ROUGH_V = 0.003      # resample jitter on velocity
PF_Z_RESAMP = 0.5
PF_Z_GR_WIN = 5           # rolling window used for the smoothed GR comparison
PF_Z_INIT_POS_STD = 0.5
PF_Z_INIT_VEL_STD = 0.02


@njit(cache=True)
def _pf_z_kernel(md_v, z_v, gr_v, gr_sm_v, grid_p, grid_s, vmin, step, gr_sigma,
                  init_pos, init_vel, beta, icpt, zsig, N,
                  MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    """Particle filter over TVT directly. Each particle's velocity is softly
    pulled toward `beta * (current Z-rate) + icpt` — the expected TVT-rate
    given how fast Z is changing, learned from the known section — in
    addition to being reweighted by GR match (both raw and smoothed)."""
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = init_pos + 0.5 * np.random.randn()
        vel[j] = init_vel + 0.02 * np.random.randn()

    n = len(md_v)
    pts = np.empty(n); stds = np.empty(n)
    prev_md = md_v[0] - 1.0
    prev_z = z_v[0] - 1.0

    for i in range(n):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        z_rate = (z_v[i] - prev_z) / dm
        expected_vel = beta * z_rate + icpt

        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            lo = vmin - 50.0
            hi = vmin + len(grid_p) * step + 50.0
            if pos[j] < lo: pos[j] = lo
            if pos[j] > hi: pos[j] = hi

        if not np.isnan(gr_v[i]):
            wsum = 0.0
            for j in range(N):
                ep = _interp1(grid_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gr_sigma
                ddp = dp * dp
                lp = np.exp(-0.5 * ddp) if ddp < 600.0 else 0.0
                if lp < 1e-300: lp = 1e-300
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(grid_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gr_sigma * 1.5)
                    dds = ds * ds
                    lsm = np.exp(-0.5 * dds) if dds < 600.0 else 0.0
                    if lsm < 1e-300: lsm = 1e-300
                    lk = (1.0 - GR_WT) * lp + GR_WT * lsm
                else:
                    lk = lp
                if lk < 1e-300: lk = 1e-300
                w[j] *= lk
                wsum += w[j]
            if wsum > 0.0:
                for j in range(N): w[j] /= wsum
            else:
                for j in range(N): w[j] = 1.0 / N

        # Velocity-physics constraint: penalize particles whose TVT-rate strays
        # from the Z-rate-implied expectation.
        wsum2 = 0.0
        vzsig = zsig * 2.0
        if vzsig < 0.005: vzsig = 0.005
        for j in range(N):
            dv = (vel[j] - expected_vel) / vzsig
            ddv = dv * dv
            lz = np.exp(-0.5 * ddv) if ddv < 600.0 else 0.0
            if lz < 1e-300: lz = 1e-300
            w[j] *= lz
            wsum2 += w[j]
        if wsum2 > 0.0:
            for j in range(N): w[j] /= wsum2
        else:
            for j in range(N): w[j] = 1.0 / N

        neff = 0.0
        for j in range(N): neff += w[j] * w[j]
        neff = 1.0 / neff
        if neff < RESAMP * N:
            pos, vel = _pf_resample(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1.0 / N

        wm = 0.0
        for j in range(N): wm += w[j] * pos[j]
        pts[i] = wm
        var = 0.0
        for j in range(N): var += w[j] * (pos[j] - wm) ** 2
        stds[i] = var ** 0.5
        prev_md = md_v[i]; prev_z = z_v[i]

    return pts, stds


def run_pf_z(h, eval_row_idx, tw_clean, last_TVT, last_MD, last_Z, visible, N=PF_Z_N, seed=SEED):
    """Run the Z-velocity-coupled particle filter over ALL hidden rows for
    this well, returning (pf_z_tvt, pf_z_std) aligned to eval_row_idx. Falls
    back to a flat last_TVT prediction (=> pf_z_delta of 0) if there's no
    usable typewell or too little known-section data to fit the velocity
    regression."""
    n = len(eval_row_idx)
    if n == 0:
        return np.array([], np.float32), np.array([], np.float32)
    if tw_clean is None or len(tw_clean) < 3 or len(visible) < 15:
        return np.full(n, last_TVT, np.float32), np.zeros(n, np.float32)

    tw_tvt = tw_clean["TVT"].to_numpy(np.float64)
    tw_gr  = tw_clean["GR"].to_numpy(np.float64)
    grid_p, vmin, step = _typewell_grid(tw_tvt, tw_gr)

    tw_gr_smooth = pd.Series(tw_gr).rolling(PF_Z_GR_WIN, center=True, min_periods=1).mean().to_numpy(np.float64)
    grid_s, _, _ = _typewell_grid(tw_tvt, tw_gr_smooth)

    kn_gr = visible["GR"].values if "GR" in visible.columns else np.array([])
    if len(kn_gr):
        expected_at_kn = np.interp(visible["TVT_input"].values, tw_tvt, tw_gr)
        gr_sigma = float(np.clip(np.nanstd(kn_gr - expected_at_kn),
                                  PF_ANCC_GR_SIG_MIN, PF_ANCC_GR_SIG_MAX))
    else:
        gr_sigma = PF_ANCC_GR_SIG_DEF

    # Regress TVT-rate on Z-rate over the whole known section: how much does
    # TVT typically move per unit of Z movement, at this well's inclination?
    dz_k  = np.diff(visible["Z"].values)
    dvt_k = np.diff(visible["TVT_input"].values)
    dmd_k = np.diff(visible["MD"].values)
    mk = dmd_k > 0
    if mk.sum() >= 10:
        vz = dz_k[mk] / dmd_k[mk]
        vt = dvt_k[mk] / dmd_k[mk]
        A = np.column_stack([vz, np.ones_like(vz)])
        coef, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt = float(coef[0]), float(coef[1])
        zsig = max(float(np.std(vt - (coef[0] * vz + coef[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1

    # Initial TVT velocity from the last 20 known rows.
    tail = visible.tail(20)
    dvt2 = np.diff(tail["TVT_input"].values)
    dmd2 = np.diff(tail["MD"].values)
    m2 = dmd2 > 0
    init_vel = float(np.median(dvt2[m2] / dmd2[m2])) if m2.sum() >= 3 else 0.0

    md_v = h["MD"].values[eval_row_idx].astype(np.float64)
    z_v  = h["Z"].values[eval_row_idx].astype(np.float64)
    gr_series = pd.Series(h["GR"].values).interpolate(limit_direction="both")
    gr_v = gr_series.values[eval_row_idx].astype(np.float64)
    gr_sm_series = pd.Series(h["GR"].values).rolling(PF_Z_GR_WIN, center=True, min_periods=1).mean()
    gr_sm_v = gr_sm_series.values[eval_row_idx].astype(np.float64)

    if seed is not None:
        _pf_seed(int(seed) + 1)  # distinct stream from pf_ancc
    pts, stds = _pf_z_kernel(md_v, z_v, gr_v, gr_sm_v, grid_p, grid_s, vmin, step, gr_sigma,
                              last_TVT, init_vel, beta, icpt, zsig, N,
                              PF_Z_MOM, PF_Z_VN, PF_Z_PN, PF_Z_GR_WT,
                              PF_Z_ROUGH_P, PF_Z_ROUGH_V, PF_Z_RESAMP)
    return pts.astype(np.float32), stds.astype(np.float32)



################ Feature engineering ################
def estimate_alignment_from_visible(hw_visible, tw):
    """Estimate the typewell shift Δ, correlation r, and bias from the visible portion.

    Uses only rows where TVT_input is present, so this is computable at inference time.
    Returns dict with keys: shift_ft, corr, bias."""
    out = {"shift_ft": 0.0, "corr": np.nan, "bias": 0.0}
    if tw is None or "TVT" not in tw or "GR" not in tw or "GR" not in hw_visible:
        return out
    tw_clean = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw_clean) < 10: return out
    sub = hw_visible.dropna(subset=["TVT_input", "GR"])
    if len(sub) < 50: return out
    tvt = sub["TVT_input"].values
    gr_h = sub["GR"].values
    best_r = -np.inf; best_s = 0.0
    shifts = np.arange(-30, 31, 2)
    for s in shifts:
        g = np.interp(tvt + s, tw_clean["TVT"].values, tw_clean["GR"].values,
                      left=np.nan, right=np.nan)
        v = np.isfinite(g) & np.isfinite(gr_h)
        if v.sum() < 30: continue
        rs = float(np.corrcoef(gr_h[v], g[v])[0, 1])
        if np.isfinite(rs) and rs > best_r:
            best_r = rs; best_s = float(s)
    # Bias at best shift
    if np.isfinite(best_r):
        g = np.interp(tvt + best_s, tw_clean["TVT"].values, tw_clean["GR"].values,
                      left=np.nan, right=np.nan)
        v = np.isfinite(g) & np.isfinite(gr_h)
        out["shift_ft"] = best_s
        out["corr"] = best_r
        out["bias"] = float(np.nanmean(gr_h[v] - g[v])) if v.any() else 0.0
    return out



def build_features_for_well(wid, split, test_eval_idx=None):
    """Construct the feature matrix for one well's evaluation rows.

    Args:
        wid: well id.
        split: 'train' or 'test'.
        test_eval_idx: for test wells, set of row indices required by the submission.
                       If None, defaults to all TVT_input == NaN rows.
    Returns: DataFrame with one row per evaluation row, including 'id', 'well_id',
             'row_index', features, and (for train) 'target_residual' and 'target_tvt'."""
    h = pd.read_csv(horiz_path(wid, split))
    h["row_index"] = np.arange(len(h), dtype=np.int64)
    if "TVT_input" not in h.columns:
        return pd.DataFrame()
    
    # Identify evaluation rows.
    eval_mask = h["TVT_input"].isna().values
    if not eval_mask.any():
        return pd.DataFrame()
    
    if split == "train":
        if "TVT" not in h.columns:
            return pd.DataFrame()
        # Use rows where the target is known and TVT_input is hidden.
        sel_mask = eval_mask & h["TVT"].notna().values
    else:
        sel_mask = eval_mask
        if test_eval_idx is not None:
            mask2 = np.zeros(len(h), dtype=bool)
            mask2[list(test_eval_idx)] = True
            sel_mask = sel_mask & mask2
    
    if not sel_mask.any():
        return pd.DataFrame()
    
    visible = h[h["TVT_input"].notna()].copy()
    if len(visible) == 0: return pd.DataFrame()
    
    last = visible.iloc[-1]
    last_TVT = float(last["TVT_input"])
    last_MD  = float(last["MD"])
    last_X   = float(last["X"])
    last_Y   = float(last["Y"])
    last_Z   = float(last["Z"])
    last_GR  = float(last["GR"]) if "GR" in last and pd.notna(last["GR"]) else np.nan
    
    ps_idx = first_nan_idx(h["TVT_input"])
    
    # Visible-segment statistics.
    vis_TVT = visible["TVT_input"].values
    vis_GR  = visible["GR"].values if "GR" in visible.columns else np.array([])
    vis_n = len(visible)
    
    # Slopes of TVT vs MD over multiple windows.
    slope_all = robust_slope(visible["MD"].values, vis_TVT)
    def slope_window(K):
        if vis_n < 2: return slope_all
        return robust_slope(visible["MD"].values[-min(K, vis_n):],
                            vis_TVT[-min(K, vis_n):], default=slope_all)
    slope_K50  = slope_window(50)
    slope_K200 = slope_window(200)
    slope_K500 = slope_window(500)
    slope_TVT_Z = robust_slope(visible["Z"].values[-min(200, vis_n):],
                               vis_TVT[-min(200, vis_n):])
    
    # Typewell load and alignment.
    tp = type_path(wid, split)
    tw = pd.read_csv(tp) if tp.is_file() else None
    align = estimate_alignment_from_visible(visible, tw)
    
    # Typewell summary statistics.
    if tw is not None and len(tw):
        tw_TVT_min = float(tw["TVT"].min()) if "TVT" in tw else np.nan
        tw_TVT_max = float(tw["TVT"].max()) if "TVT" in tw else np.nan
        tw_GR_med  = float(tw["GR"].median()) if "GR" in tw else np.nan
        tw_GR_std  = float(tw["GR"].std()) if "GR" in tw else np.nan
        tw_clean = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT") if ("TVT" in tw and "GR" in tw) else None
    else:
        tw_TVT_min = tw_TVT_max = tw_GR_med = tw_GR_std = np.nan
        tw_clean = None
    
    # Per-well GR normalisation parameters.
    pw_gr_med = float(np.nanmedian(vis_GR)) if len(vis_GR) else np.nan
    pw_gr_std = float(np.nanstd(vis_GR)) if len(vis_GR) else np.nan
    if not (pw_gr_std > 0): pw_gr_std = 1.0
    
    # Build per-row feature frame.
    sel_idx = np.flatnonzero(sel_mask)
    cur = h.iloc[sel_idx].copy()
    cur["well_id"] = wid
    cur["id"] = cur["well_id"] + "_" + cur["row_index"].astype(str)
    
    # Static per-well features (broadcast).
    static = {
        "last_known_TVT": last_TVT,
        "last_known_MD": last_MD, 
        "last_known_X": last_X, 
        "last_known_Y": last_Y,
        "last_known_Z": last_Z, 
        "last_known_GR": last_GR,
        "vis_n_rows": vis_n,
        "vis_md_range": float(visible["MD"].max() - visible["MD"].min()),
        "vis_TVT_min": float(np.nanmin(vis_TVT)),
        "vis_TVT_max": float(np.nanmax(vis_TVT)),
        "vis_TVT_range": float(np.nanmax(vis_TVT) - np.nanmin(vis_TVT)),
        "vis_TVT_mean": float(np.nanmean(vis_TVT)),
        "vis_TVT_std":  float(np.nanstd(vis_TVT)),
        "vis_TVT_first": float(vis_TVT[0]),
        "vis_GR_mean":  float(np.nanmean(vis_GR)) if len(vis_GR) else np.nan,
        "vis_GR_std":   pw_gr_std,
        "vis_GR_median": pw_gr_med,
        "vis_GR_min":   float(np.nanmin(vis_GR)) if len(vis_GR) else np.nan,
        "vis_GR_max":   float(np.nanmax(vis_GR)) if len(vis_GR) else np.nan,
        "slope_TVT_MD_all":  slope_all,
        "slope_TVT_MD_K50":  slope_K50,
        "slope_TVT_MD_K200": slope_K200,
        "slope_TVT_MD_K500": slope_K500,
        "slope_TVT_Z_K200":  slope_TVT_Z,
        "tw_TVT_min": tw_TVT_min, "tw_TVT_max": tw_TVT_max,
        "tw_GR_med": tw_GR_med, "tw_GR_std": tw_GR_std,
        "align_shift_ft": align["shift_ft"],
        "align_corr": align["corr"],
        "align_bias": align["bias"],
        "ps_row_idx": ps_idx if ps_idx is not None else len(h),
        "n_total_rows": len(h),
    }
    for k, v in static.items():
        cur[k] = v
    
    # Per-row dynamic features.
    cur["md_from_ps"] = (cur["MD"].values - last_MD).astype(np.float32)
    cur["z_from_ps"] = (cur["Z"].values - last_Z).astype(np.float32)
    cur["x_from_ps"] = (cur["X"].values - last_X).astype(np.float32)
    cur["y_from_ps"] = (cur["Y"].values - last_Y).astype(np.float32)
    cur["dxy_from_ps"] = np.sqrt((cur["X"].values - last_X)**2 + (cur["Y"].values - last_Y)**2)
    cur['dxyz_from_ps'] = np.sqrt((cur["X"].values - last_X)**2 + 
                              (cur["Y"].values - last_Y)**2 +
                              (cur["Z"].values - last_Z)) # CZ:20260701 
    cur["row_from_ps"] = cur["row_index"].values - (ps_idx if ps_idx is not None else 0)
    cur["row_frac"]    = cur["row_index"].values / max(len(h) - 1, 1)
    cur["GR_norm"]     = (cur["GR"].values - pw_gr_med) / pw_gr_std
    
    # Linear projection using the local (last-50) slope: if TVT kept moving at
    # slope_K50 ft/ft past the anchor, this is the predicted TVT delta at each
    # row. Equivalent to `slp_b_d_50` in https://www.kaggle.com/code/damiendesolei/rogii-lb7156-baseline-visualization?scriptVersionId=333120594
    cur["slp_b_d_50"] = slope_K50 * cur["md_from_ps"].values
    #cur["slp_b_200"] = slope_K200 * cur["md_from_ps"].values
    #cur["slp_b_500"] = slope_K500 * cur["md_from_ps"].values
    cur["slp_b_d_all"] = slope_all * cur["md_from_ps"].values
    
    
    # Spatial ANCC imputation: KNN over sampled points from OTHER training
    # wells. `dense_dist` is the distance to the nearest neighbour actually
    # used, i.e. how spatially isolated this evaluation point is from wells
    # with known ANCC (large => imputation is extrapolating, less trustworthy).
    xy_ev = cur[["X", "Y"]].to_numpy(np.float64)
    self_wid = wid if split == "train" else None
    dense_ancc, dense_std, dense_dist = DI.impute(xy_ev, self_wid=self_wid)
    cur["dense_ancc"] = dense_ancc
    cur["dense_std"]  = dense_std
    cur["dense_dist"] = dense_dist
    
    # Well-level formation plane fit (distinct from the point-level DenseANCCImputer
    # above): each well collapses to one representative point, and spatial_knn_dist
    # is the distance to the nearest neighbouring well used in that plane fit.
    _, spatial_knn_dist = FI.impute(xy_ev, self_wid=self_wid)
    cur["spatial_knn_dist"] = spatial_knn_dist

    # Particle-filter TVT tracker (ANCC-anchored). Run once over the FULL hidden
    # region (all rows past the anchor, in order) so the simulation's momentum
    # is continuous, then subset down to this well's selected rows.
    eval_row_idx = np.flatnonzero(eval_mask)
    pf_full, pf_std_full = run_pf_ancc(h, eval_row_idx, tw_clean, last_TVT, last_MD, last_Z, visible)
    pos_in_eval = np.searchsorted(eval_row_idx, sel_idx)
    cur["pf_ancc"] = pf_full[pos_in_eval]
    cur["pf_ancc_std"] = pf_std_full[pos_in_eval]
    cur["pf_ancc_delta"] = (cur["pf_ancc"].values - last_TVT).astype(np.float32)

    # Second, independent tracker: TVT directly, velocity-constrained against
    # the well's Z-rate. Falls back to a flat last_TVT (=> zero delta) when
    # there isn't enough known-section data to fit the velocity regression.
    pf_z_full, pf_z_std_full = run_pf_z(h, eval_row_idx, tw_clean, last_TVT, last_MD, last_Z, visible)
    cur["pf_z"] = pf_z_full[pos_in_eval]
    cur["pf_z_std"] = pf_z_std_full[pos_in_eval]
    cur["pf_z_delta"] = (cur["pf_z"].values - last_TVT).astype(np.float32)
    # Disagreement between the two independent trackers — large values flag
    # rows where the GR-only tracker and the Z-constrained tracker diverge.
    cur["pf_vs_z"] = (cur["pf_ancc"].values - cur["pf_z"].values).astype(np.float32)
    
    # GR comparison with typewell at last_known_TVT − Δ (a fixed depth proxy).
    if tw_clean is not None and len(tw_clean):
        gr_at_last = float(np.interp(last_TVT - align["shift_ft"],
                                      tw_clean["TVT"].values, tw_clean["GR"].values,
                                      left=np.nan, right=np.nan))
        cur["gr_minus_tw_at_last"] = cur["GR"].values - gr_at_last - align["bias"]
        # Best-match TVT within ±150 ft of last_known_TVT.
        win = tw_clean[(tw_clean["TVT"] >= last_TVT - 150) &
                       (tw_clean["TVT"] <= last_TVT + 150)]
        if len(win) >= 5:
            tvts = win["TVT"].values
            grs  = win["GR"].values
            grs_db = grs + align["bias"]  # adjust typewell GR for bias
            best_match = np.empty(len(cur))
            for i, g in enumerate(cur["GR"].values):
                if not np.isfinite(g):
                    best_match[i] = last_TVT
                else:
                    j = int(np.argmin(np.abs(grs_db - g)))
                    best_match[i] = tvts[j]
            cur["best_match_tvt"] = best_match
            cur["best_match_delta"] = best_match - last_TVT
        else:
            cur["best_match_tvt"] = last_TVT
            cur["best_match_delta"] = 0.0
    else:
        cur["gr_minus_tw_at_last"] = np.nan
        cur["best_match_tvt"] = last_TVT
        cur["best_match_delta"] = 0.0
    
    # Centred rolling GR statistics over all rows of the well (uses eval-zone GR too,
    # which is available at inference). Window = 25 rows.
    
    # GR before interpolation  CZ:20260704
    gr_full_original = h["GR"].values
    gr_series_original = pd.Series(gr_full_original)
    roll_mean_original = gr_series_original.rolling(25, min_periods=1).mean().values
    roll_std_original = gr_series_original.rolling(25, min_periods=1).std().fillna(0.0).values
    cur["roll_GR_orig_mean25"] = roll_mean_original[sel_idx]
    cur["roll_GR_orig_std25"]  = roll_std_original[sel_idx]
    cur["roll_GR_orig_mean25_diff"] = cur["roll_GR_orig_mean25"].diff()
    
    # GR after interpolation
    gr_full = h["GR"].values
    gr_series = pd.Series(gr_full).interpolate(limit_direction="both")
    roll_mean = gr_series.rolling(25, center=True, min_periods=1).mean().values
    roll_std  = gr_series.rolling(25, center=True, min_periods=1).std().fillna(0.0).values
    cur["roll_GR_mean25"] = roll_mean[sel_idx]
    cur["roll_GR_std25"]  = roll_std[sel_idx]
    cur["GR_minus_rollmean"] = cur["GR"].values - cur["roll_GR_mean25"].values
    
    # Persistence-anchor target.
    if split == "train":
        cur["target_tvt"] = h["TVT"].values[sel_idx]
        cur["target_residual"] = cur["target_tvt"].values - last_TVT
    
    return cur.reset_index(drop=True)





################ Build training matrix ################
print("Building spatial ANCC imputer from training wells...")
DI = DenseANCCImputer(train_wells, TRAIN_DIR)
print(f"  index built from {len(DI.ancc):,} sample points across "
      f"{len(set(DI.wids)):,} wells")

print("Building formation-plane KNN imputer from training wells...")
FI = FormationPlaneKNN(train_wells, TRAIN_DIR)
print(f"  index built from {len(FI.df):,} wells")

t0 = time.time()
train_parts = []
for k, wid in enumerate(train_wells):
    df = build_features_for_well(wid, "train")
    if len(df):
        train_parts.append(df)
    if (k + 1) % 100 == 0:
        print(f"  built features for {k+1}/{len(train_wells)} wells "
              f"(rows so far: {sum(len(p) for p in train_parts):,})")
train_df = pd.concat(train_parts, ignore_index=True)
del train_parts; gc.collect()
print(f"\nTrain feature matrix: {train_df.shape}  (built in {time.time()-t0:.1f}s)")
print(train_df.head(2))
train_df.to_csv("train_df.csv", index=False)


# Define the feature column list.
# Exclude IDs, targets, and any column not present at inference time.
# - TVT is the actual target (would be a direct leak in training).
# - TVT_input is all-NaN in selected rows by construction (eval-zone rows).
# - Formation columns (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) exist only in train.
exclude = {
    "well_id", "id", "row_index", "target_tvt", "target_residual",
    "TVT", "TVT_input",
    "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA",
}
feature_cols = [c for c in train_df.columns if c not in exclude]
print(f"Number of features: {len(feature_cols)}")
print("First 20 features:", feature_cols[:20])

# Cast to float32 for memory and speed.
X_train = train_df[feature_cols].astype(np.float32).values
y_train = train_df["target_residual"].astype(np.float32).values
groups  = train_df["well_id"].values
print(f"X shape: {X_train.shape}  y shape: {y_train.shape}")




################ Build test matrix ################
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Distinct test wells:     {sample_sub['well_id'].nunique()}")

test_eval_index = (sample_sub.groupby("well_id")["row_index"]
                   .apply(lambda s: set(s.tolist()))
                   .to_dict())

t0 = time.time()
test_parts = []
for k, wid in enumerate(test_wells):
    df = build_features_for_well(wid, "test", test_eval_idx=test_eval_index.get(wid))
    if len(df):
        test_parts.append(df)
test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
print(f"\nTest feature matrix: {test_df.shape}  (built in {time.time()-t0:.1f}s)")




############### Hyper ##############
gkf = GroupKFold(n_splits=3)
# Precompute the folds once so every trial uses the identical split.
cv_folds = list(gkf.split(X_train, y_train, groups=groups))

def objective(trial):
    # Define parameter search space
    param = {
        "objective": "regression",
        #"n_estimators": trial.suggest_categorical("n_estimators", [500, 1000, 1500, 2000])
        "n_estimators": 5000,
        "metric": "rmse",  
        #"boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "feature_fraction": trial.suggest_categorical("feature_fraction", [0.8, 0.85, 0.9, 0.95]),
        "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.8, 0.85, 0.9, 0.95]),
        "bagging_freq": trial.suggest_int("bagging_freq", 5, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 12, 256),
        "max_depth": trial.suggest_int("max_depth", 2, 32),  # -1 means no limit
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        #"min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        "device_type": "gpu",  # Enable GPU support
        "verbosity": -1,
        "seed" : SEED

    }

    fold_rmse_tvt = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        assert tr_idx.max() < X_train.shape[0] and tr_idx.min() >= 0, \
           f"corrupt tr_idx at fold {fold}: max={tr_idx.max()}, min={tr_idx.min()}, X_train rows={X_train.shape[0]}"
        assert va_idx.max() < X_train.shape[0] and va_idx.min() >= 0, \
           f"corrupt va_idx at fold {fold}: max={va_idx.max()}, min={va_idx.min()}"
        dtrain = lgb.Dataset(X_train[tr_idx], y_train[tr_idx], feature_name=feature_cols)
        dvalid = lgb.Dataset(X_train[va_idx], y_train[va_idx], feature_name=feature_cols, reference=dtrain)

        model = lgb.train(
            param, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(10)],
        )
        pred_residual = model.predict(X_train[va_idx], num_iteration=model.best_iteration)

        # Reconstruct TVT from the persistence anchor, same as your CV loop.
        y_true_tvt = train_df["target_tvt"].values[va_idx]
        last_tvt   = train_df["last_known_TVT"].values[va_idx]
        pred_tvt   = last_tvt + pred_residual

        rmse_tvt = float(np.sqrt(np.mean((pred_tvt - y_true_tvt) ** 2)))
        fold_rmse_tvt.append(rmse_tvt)

        trial.report(np.mean(fold_rmse_tvt), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_rmse_tvt))
 

# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(
    study_name="lgb_TVT_tuning_20260712_2",
    storage="sqlite:///lgb_TVT_tuning.db" ,
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, timeout=9*3600, n_jobs=5) # n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best average rmse:", study.best_value)


# Get the best parameters and score
#best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
#file_name = f"lgb_rogii_parameters_{best_score:.6f}.csv"

# Save the best parameters to a CSV file
#df_param = pd.DataFrame([best_params])  # Convert to DataFrame
#df_param.to_csv(file_name, index=False)  # Save to CSV
study.trials_dataframe().to_csv(f"lgb_rogii_{best_score:.6f}.csv", index=False)
#print(f"Best parameters saved to {file_name}")

#### Check optuna results ###
s = optuna.load_study(study_name="lgb_TVT_tuning_20260712_2", storage="sqlite:///lgb_TVT_tuning.db")
print(s.best_value, s.best_params)
#s.trials_dataframe().tail(10) 



################ CV ################
gkf = GroupKFold(n_splits=N_FOLDS)
oof_residual = np.zeros(len(train_df), dtype=np.float64)
fold_models = []
fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    print(f"  train rows: {len(tr_idx):,}  val rows: {len(va_idx):,}  "
          f"train wells: {pd.Series(groups[tr_idx]).nunique()}  val wells: {pd.Series(groups[va_idx]).nunique()}")
    
    dtrain = lgb.Dataset(X_train[tr_idx], y_train[tr_idx], feature_name=feature_cols)
    dvalid = lgb.Dataset(X_train[va_idx], y_train[va_idx], feature_name=feature_cols,
                          reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    pred_residual = booster.predict(X_train[va_idx], num_iteration=booster.best_iteration)
    oof_residual[va_idx] = pred_residual
    
    # Per-fold RMSE on TVT (residual + persistence anchor).
    y_true_tvt = train_df["target_tvt"].values[va_idx]
    last_tvt = train_df["last_known_TVT"].values[va_idx]
    pred_tvt = last_tvt + pred_residual
    rmse_residual_only = float(np.sqrt(np.mean((pred_residual - y_train[va_idx])**2)))
    rmse_tvt = float(np.sqrt(np.mean((pred_tvt - y_true_tvt)**2)))
    rmse_b0 = float(np.sqrt(np.mean((last_tvt - y_true_tvt)**2)))
    print(f"  fold RMSE  residual-only = {rmse_residual_only:.4f}")
    print(f"  fold RMSE  TVT (model)   = {rmse_tvt:.4f}")
    print(f"  fold RMSE  TVT (B0 only) = {rmse_b0:.4f}")
    
    fold_models.append(booster)
    fold_metrics.append({"fold": fold + 1, "best_iter": booster.best_iteration,
                          "rmse_residual": rmse_residual_only,
                          "rmse_tvt_model": rmse_tvt, "rmse_tvt_B0": rmse_b0})

print()
print("Per-fold summary:")
print(pd.DataFrame(fold_metrics).round(4))




# Overall OOF metrics.
last_tvt_all = train_df["last_known_TVT"].values
oof_tvt = last_tvt_all + oof_residual
y_true_all = train_df["target_tvt"].values
rmse_oof_tvt = float(np.sqrt(np.mean((oof_tvt - y_true_all)**2)))
rmse_oof_b0  = float(np.sqrt(np.mean((last_tvt_all - y_true_all)**2)))
print(f"OOF RMSE — model: {rmse_oof_tvt:.4f}")
print(f"OOF RMSE — B0:    {rmse_oof_b0:.4f}")
print(f"Improvement over B0: {rmse_oof_b0 - rmse_oof_tvt:+.4f} ft "
      f"({(1 - rmse_oof_tvt / rmse_oof_b0) * 100:+.2f}%)")


# Per-well RMSE distribution.
per_well = (pd.DataFrame({"well_id": train_df["well_id"].values,
                           "y_true": y_true_all, "y_pred": oof_tvt,
                           "y_b0":   last_tvt_all})
            .groupby("well_id")
            .apply(lambda g: pd.Series({
                "rmse_model": float(np.sqrt(np.mean((g["y_pred"] - g["y_true"])**2))),
                "rmse_b0":    float(np.sqrt(np.mean((g["y_b0"]   - g["y_true"])**2))),
                "n":          len(g)}))
            .reset_index())
per_well["delta_vs_b0"] = per_well["rmse_b0"] - per_well["rmse_model"]
print("Per-well RMSE summary:")
print(per_well[["rmse_model", "rmse_b0", "delta_vs_b0", "n"]].describe(percentiles=[0.1, 0.5, 0.9]).round(3))
print(f"\nWells where the model beats B0: {(per_well['delta_vs_b0'] > 0).sum()} / {len(per_well)} "
      f"({(per_well['delta_vs_b0'] > 0).mean() * 100:.1f}%)")




# Histogram and scatter.
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
ax[0].hist(per_well["delta_vs_b0"], bins=50, color="#1f77b4", edgecolor="white")
ax[0].axvline(0, color="black", lw=0.7, ls="--")
ax[0].set_title("Per-well RMSE improvement vs B0 (positive = model better)")
ax[0].set_xlabel("RMSE(B0) − RMSE(model)  [ft]")
ax[0].set_ylabel("number of wells")

ax[1].scatter(per_well["rmse_b0"], per_well["rmse_model"], s=8, alpha=0.5)
lims = [0, max(per_well["rmse_b0"].max(), per_well["rmse_model"].max())]
ax[1].plot(lims, lims, color="black", lw=0.7, ls="--", label="y = x")
ax[1].set_xlabel("RMSE — B0 (ft)")
ax[1].set_ylabel("RMSE — model (ft)")
ax[1].set_title("Per-well RMSE: model vs B0")
ax[1].legend()
plt.show()




# Feature importance averaged over folds.
imp_arr = np.zeros(len(feature_cols))
for booster in fold_models:
    imp_arr += booster.feature_importance(importance_type="gain")
imp_arr /= len(fold_models)
imp_df = (pd.DataFrame({"feature": feature_cols, "gain": imp_arr})
          .sort_values("gain", ascending=False).reset_index(drop=True))
print("Top 25 features by mean gain:")
print(imp_df.head(25))

fig, ax = plt.subplots(figsize=(8, 8))
top = imp_df.head(25)[::-1]
ax.barh(top["feature"], top["gain"], color="#1f77b4")
ax.set_title("LightGBM feature importance — top 25 (mean gain over folds)")
ax.set_xlabel("gain")
plt.show()



### Test prediction and submission
if len(test_df):
    X_test = test_df[feature_cols].astype(np.float32).values
    fold_preds = np.zeros((len(test_df), len(fold_models)), dtype=np.float64)
    for j, booster in enumerate(fold_models):
        fold_preds[:, j] = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred_residual = fold_preds.mean(axis=1)
    
    # Clip residual to a defensible range based on training data.
    residual_clip = 400.0
    pred_residual = np.clip(pred_residual, -residual_clip, residual_clip)
    
    pred_tvt = test_df["last_known_TVT"].values + pred_residual
    test_df["tvt_pred"] = pred_tvt
    print(f"Predicted residual stats — mean: {pred_residual.mean():+.3f}  "
          f"std: {pred_residual.std():.3f}  "
          f"min: {pred_residual.min():.3f}  max: {pred_residual.max():.3f}")
    print(f"Predicted TVT stats     — mean: {pred_tvt.mean():.1f}  "
          f"min: {pred_tvt.min():.1f}  max: {pred_tvt.max():.1f}")
else:
    print("Test feature matrix is empty.")
    
    

# Build the submission file.
if len(test_df):
    pred_map = test_df.set_index("id")["tvt_pred"].to_dict()
    submission = sample_sub[["id"]].copy()
    submission["tvt"] = submission["id"].map(pred_map)
    
    # Fill any missing predictions with the well's last_known_TVT (defensive fallback).
    if submission["tvt"].isna().any():
        anchor_map = test_df.drop_duplicates("well_id").set_index("well_id")["last_known_TVT"].to_dict()
        miss = submission["tvt"].isna()
        submission.loc[miss, "tvt"] = (
            submission.loc[miss, "id"].str.rsplit("_", n=1).str[0].map(anchor_map)
        )
    submission["tvt"] = submission["tvt"].fillna(0.0)
    
    submission.to_csv("submission.csv", index=False)
    print(f"Submission written: shape = {submission.shape}")
    print(submission.head())
    print()
    print("tvt summary in submission:")
    print(submission["tvt"].describe().round(3))
else:
    # Defensive: still produce a valid submission file using last-known TVT only.
    submission = sample_sub[["id"]].copy()
    submission["tvt"] = 0.0
    submission.to_csv("submission.csv", index=False)
    print("Test set was empty; wrote zero-filled submission.")