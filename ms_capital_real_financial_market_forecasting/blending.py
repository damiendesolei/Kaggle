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
# candidate weight sets (lgb, cat, mlp); submit a few and compare on LB
CANDIDATES = {
    "eq_111":   {"lgb": 1.0, "cat": 1.0, "mlp": 1.0},
    "mlp_2x":   {"lgb": 1.0, "cat": 1.0, "mlp": 2.0},
    "mlp_4x":   {"lgb": 1.0, "cat": 1.0, "mlp": 4.0},
    "mlp_heavy": {"lgb": 0.5, "cat": 0.5, "mlp": 4.0},
    "mlp_lgb":  {"lgb": 1.0, "cat": 0.0, "mlp": 2.0},
    "mlp_cat":  {"lgb": 0.0, "cat": 1.0, "mlp": 2.0},
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
names = list(P)

# ------------------------------------------------------------------
# Sanity checks: are the files really different?
# ------------------------------------------------------------------
print("n rows:", len(ids), "| NaNs:", {k: int(np.isnan(v).sum()) for k, v in P.items()})
print("pred mean/std:", {k: (round(float(v.mean()), 6), round(float(v.std()), 6)) for k, v in P.items()})
C = np.corrcoef(np.vstack([P[k] for k in names]))
print("pairwise corr:")
print(pl.DataFrame({"model": names, **{n: C[:, i] for i, n in enumerate(names)}}))

# ------------------------------------------------------------------
# Build each candidate and report how close it is to each single model
# ------------------------------------------------------------------
for tag, W in CANDIDATES.items():
    wsum = sum(W.values())
    blend = sum(W[k] * z(P[k]) for k in names) / wsum
    cors = {k: round(float(np.corrcoef(blend, P[k])[0, 1]), 4) for k in names}
    print(f"{tag:10s} corr(blend, model): {cors}")
    pl.DataFrame({
        "sample_id": pl.Series(ids, dtype=pl.Int32),
        "prediction": pl.Series(blend, dtype=pl.Float64),
    }).write_csv(f"blend_{tag}.csv")
print("saved blend_*.csv")
print(f"saved {OUT_CSV}")