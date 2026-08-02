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


############### Lasso Hyper-parameter Tuning ##############
# Lasso needs a) no NaNs and b) comparably-scaled features, neither of which
# LightGBM cares about, so each trial fits a small Imputer -> Scaler -> Lasso
# pipeline. Reuses the same `cv_folds` (and TVT reconstruction) as the
# LightGBM objective above so the two models are compared on identical splits.
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


############### Hyper ##############
gkf = GroupKFold(n_splits=3)
# Precompute the folds once so every trial uses the identical split.
cv_folds = list(gkf.split(X, y, groups=g))

def Lasso_objective(trial):
    param = {
        "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
        "imputer_strategy": trial.suggest_categorical("imputer_strategy", ["mean", "median"]),
        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "fit_intercept": True
    }

    fold_rmse_tvt = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=param["imputer_strategy"])),
            ("scaler", RobustScaler()),
            ("Lasso", Lasso(alpha=param["alpha"],
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


# Run Optuna study for Lasso
print("Start running Lasso hyper parameter tuning..")
Lasso_study = optuna.create_study(
    study_name="Lasso_TVT_tuning_20260728",
    storage="sqlite:///Lasso_TVT_tuning.db",
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
Lasso_study.optimize(Lasso_objective, timeout=0.5*3600, n_jobs=1)  

# Print the best hyperparameters and score
print("Best Lasso hyperparameters:", Lasso_study.best_params)
print("Best Lasso average rmse:", Lasso_study.best_value)

Lasso_best_score = Lasso_study.best_value
Lasso_study.trials_dataframe().to_csv(f"Lasso_rogii_{Lasso_best_score:.6f}.csv", index=False)

#### Check Lasso optuna results ###
rs = optuna.load_study(study_name="Lasso_TVT_tuning_20260728", storage="sqlite:///Lasso_TVT_tuning.db")
print(rs.best_value, rs.best_params)
#rs.trials_dataframe().tail(10)

Lasso_PARAMS = dict(rs.best_params)



############### Feature Importance (|coefficient|) ##############
# Refit on the FULL training set with the best trial's hyperparameters so the
# importances reflect one model trained on all rows, not just one CV fold.
final_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy=Lasso_PARAMS["imputer_strategy"])),
    ("scaler", RobustScaler()),
    ("Lasso", Lasso(alpha=Lasso_PARAMS["alpha"],
                     fit_intercept=True,
                     random_state=SEED)),
])
final_pipe.fit(X, y)
 
lasso_coefs = final_pipe.named_steps["Lasso"].coef_
n_nonzero = int(np.sum(lasso_coefs != 0))
print(f"Lasso kept {n_nonzero}/{len(features)} features nonzero at alpha={Lasso_PARAMS['alpha']:.5g}")
 
feature_importance = pd.DataFrame({
    "feature": features,
    "coefficient": lasso_coefs,
    "abs_coefficient": np.abs(lasso_coefs),
}).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
 
print("Top 20 features by |coefficient|:")
print(feature_importance.head(20).to_string(index=False))
 
feature_importance.to_csv(f"Lasso_feature_importance_{Lasso_best_score:.6f}.csv", index=False)
 
# Bar chart of the top N features, largest |coefficient| at the top.
import matplotlib.pyplot as plt
 
TOP_N = 25
plot_df = feature_importance.head(TOP_N).iloc[::-1]
colors = np.where(plot_df["coefficient"] >= 0, "tab:blue", "tab:red")
 
plt.figure(figsize=(8, max(4, 0.3 * TOP_N)))
plt.barh(plot_df["feature"], plot_df["abs_coefficient"], color=colors)
plt.xlabel("|coefficient|")
plt.title(f"Top {TOP_N} Lasso feature importances (blue = positive, red = negative)")
plt.tight_layout()
plt.savefig(f"Lasso_feature_importance_{Lasso_best_score:.6f}.png", dpi=150)
plt.show()