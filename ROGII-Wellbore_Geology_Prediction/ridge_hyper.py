# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:01:33 2026

@author: azz
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler

SEED = 42 

DATA_ROOT = Path(r"H:\kaggle\rogii-wellbore-geology-prediction")
DF = "train_df.csv"

train_df = pd.read_csv(DATA_ROOT / DF, dtype={"well": "string"})


features = [c for c in train_df.columns if c not in {'well','id','target'}]

X = train_df[features].values
y = train_df['target'].values
g = train_df['well'].values


############### Ridge Hyper-parameter Tuning ##############
# Ridge needs a) no NaNs and b) comparably-scaled features, neither of which
# LightGBM cares about, so each trial fits a small Imputer -> Scaler -> Ridge
# pipeline. Reuses the same `cv_folds` (and TVT reconstruction) as the
# LightGBM objective above so the two models are compared on identical splits.
from sklearn.linear_model import Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


############### Hyper ##############
gkf = GroupKFold(n_splits=5)
# Precompute the folds once so every trial uses the identical split.
cv_folds = list(gkf.split(X, y, groups=g))

def ridge_objective(trial):
    param = {
        "alpha": trial.suggest_float("alpha", 1e2, 1e5, log=True),
        "imputer_strategy": trial.suggest_categorical("imputer_strategy", ["mean", "median"]),
        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "fit_intercept": True
    }

    fold_rmse_tvt = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=param["imputer_strategy"])),
            ("scaler", RobustScaler()),
            ("ridge", Ridge(alpha=param["alpha"],
                             fit_intercept=param["fit_intercept"],
                             random_state=SEED)),
        ])
        pipe.fit(X[tr_idx], y[tr_idx])
        pred_residual = pipe.predict(X[va_idx])

        # Reconstruct TVT from the persistence anchor, same as the LightGBM objective.
        y_true_tvt = train_df["target"].values[va_idx]
        last_tvt   = train_df["last_known_tvt"].values[va_idx]
        #pred_tvt   = last_tvt + pred_residual
        pred_tvt   = pred_residual

        rmse_tvt = float(np.sqrt(np.mean((pred_tvt - y_true_tvt) ** 2)))
        fold_rmse_tvt.append(rmse_tvt)

        trial.report(np.mean(fold_rmse_tvt), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_rmse_tvt))


# Run Optuna study for Ridge
print("Start running ridge hyper parameter tuning..")
ridge_study = optuna.create_study(
    study_name="ridge_TVT_tuning_20260730",
    storage="sqlite:///ridge_TVT_tuning.db",
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
ridge_study.optimize(ridge_objective, timeout=0.5*3600, n_jobs=1)  

# Print the best hyperparameters and score
print("Best ridge hyperparameters:", ridge_study.best_params)
print("Best ridge average rmse:", ridge_study.best_value)

ridge_best_score = ridge_study.best_value
ridge_study.trials_dataframe().to_csv(f"ridge_rogii_{ridge_best_score:.6f}.csv", index=False)

#### Check Ridge optuna results ###
rs = optuna.load_study(study_name="ridge_TVT_tuning_20260730", storage="sqlite:///ridge_TVT_tuning.db")
print(rs.best_value, rs.best_params)
#rs.trials_dataframe().tail(10)

RIDGE_PARAMS = dict(rs.best_params)