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

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)

RNG = np.random.default_rng(42)


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
    seed=42,
    device_type='gpu'
)
NUM_BOOST_ROUND = 2000
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
        "last_known_MD": last_MD, "last_known_X": last_X, "last_known_Y": last_Y,
        "last_known_Z": last_Z, "last_known_GR": last_GR,
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
    cur["md_from_ps"] = cur["MD"].values - last_MD
    cur["z_from_ps"]  = cur["Z"].values - last_Z
    cur["dxy_from_ps"] = np.sqrt((cur["X"].values - last_X)**2 + (cur["Y"].values - last_Y)**2)
    cur['dxyz_from_ps'] = np.sqrt((cur["X"].values - last_X)**2 + 
                              (cur["Y"].values - last_Y)**2 +
                              (cur["Z"].values - last_Z)) # CZ:20260701 
    cur["row_from_ps"] = cur["row_index"].values - (ps_idx if ps_idx is not None else 0)
    cur["row_frac"]    = cur["row_index"].values / max(len(h) - 1, 1)
    cur["GR_norm"]     = (cur["GR"].values - pw_gr_med) / pw_gr_std
    
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