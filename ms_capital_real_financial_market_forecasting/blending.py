# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:36:56 2026

@author: azz
"""

import numpy as np
import polars as pl

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FILES = {
    "lgb": "lgb_submission_144572.csv",
    "cat": "catboost_submission_144387.csv",
    "mlp": "realmlp_submission_146082.csv",   # <- change to your RealMLP file name
}

# correlation
dfs = {k: pl.read_csv(v).sort("sample_id") for k, v in FILES.items()}
ids = dfs["lgb"]["sample_id"].to_numpy()
for k, d in dfs.items():
    assert (d["sample_id"].to_numpy() == ids).all(), f"sample_id mismatch in {k}"
P = {k: d["prediction"].to_numpy().astype(np.float64) for k, d in dfs.items()}
names = list(P)

# Pearson (this is what your centered-cosine metric measures)
C = np.corrcoef(np.vstack([P[k] for k in names]))
print("Pearson corr:")
print(pl.DataFrame({"model": names, **{n: C[:, i] for i, n in enumerate(names)}}))

# Spearman (rank correlation), useful to see if outliers are distorting Pearson
R = np.corrcoef(np.vstack([np.argsort(np.argsort(P[k])) for k in names]))
print("\nSpearman corr:")
print(pl.DataFrame({"model": names, **{n: R[:, i] for i, n in enumerate(names)}}))


GBM_FRACS = [0.05, 0.10]   # share of the blend given to the GBM component

def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)

# ------------------------------------------------------------------
# Load + align
# ------------------------------------------------------------------
dfs = {k: pl.read_csv(v).sort("sample_id") for k, v in FILES.items()}
ids = dfs["lgb"]["sample_id"].to_numpy()
for k, d in dfs.items():
    assert (d["sample_id"].to_numpy() == ids).all(), f"sample_id mismatch in {k}"
P = {k: d["prediction"].to_numpy().astype(np.float64) for k, d in dfs.items()}

# ------------------------------------------------------------------
# Treat lgb + cat as ONE component (they are 0.96 correlated)
# ------------------------------------------------------------------
gbm = z((z(P["lgb"]) + z(P["cat"])) / 2)
mlp = z(P["mlp"])

# ------------------------------------------------------------------
# Build and save the small-weight blends
# ------------------------------------------------------------------
for f in GBM_FRACS:
    blend = (1 - f) * mlp + f * gbm
    tag = f"mlp_gbm{int(round(f * 100)):02d}"          # -> mlp_gbm05, mlp_gbm10
    out = f"blend_{tag}.csv"
    pl.DataFrame({
        "sample_id": pl.Series(ids, dtype=pl.Int32),
        "prediction": pl.Series(blend, dtype=pl.Float64),
    }).write_csv(out)
    print(f"{out}: corr(blend, mlp) = {np.corrcoef(blend, P['mlp'])[0, 1]:.5f}")