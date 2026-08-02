# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 2026

@author: azz
"""

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import optuna
from optuna.samplers import TPESampler
from koolbox import Trainer

SEED = 42

DATA_ROOT = Path(r"H:\kaggle\rogii-wellbore-geology-prediction")
DF = "train_df.csv"

train_df = pd.read_csv(DATA_ROOT / DF, dtype={"well": "string"})
print(train_df.memory_usage(deep=True).sum() / 1e6, "MB")  # sanity check

float_cols = train_df.select_dtypes(include="float64").columns
train_df[float_cols] = train_df[float_cols].astype("float32")
print(train_df.memory_usage(deep=True).sum() / 1e6, "MB")  # sanity check

features = [c for c in train_df.columns if c not in {'well', 'id', 'target'}]

X = train_df[features].values
y = train_df['target'].values
g = train_df['well'].values


############### Hyper ##############
gkf = GroupKFold(n_splits=5)
# Precompute the folds once so every trial uses the identical split.
cv_folds = list(gkf.split(X, y, groups=g))


NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 250


def objective(trial):
    # Define parameter search space
    param = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": NUM_BOOST_ROUND,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "depth": trial.suggest_int("depth", 4, 12),#16),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),#, 254]),
        # NOTE: min_data_in_leaf is not tunable on GPU with the default
        # SymmetricTree grow policy, so it's intentionally left out here.
        "task_type": "GPU",  # Enable GPU support
        "devices": "0",
        "verbose": False,
        "random_seed": SEED,
    }

    fold_rmse_tvt = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        assert tr_idx.max() < X.shape[0] and tr_idx.min() >= 0, \
            f"corrupt tr_idx at fold {fold}: max={tr_idx.max()}, min={tr_idx.min()}, X rows={X.shape[0]}"
        assert va_idx.max() < X.shape[0] and va_idx.min() >= 0, \
            f"corrupt va_idx at fold {fold}: max={va_idx.max()}, min={va_idx.min()}"

        train_pool = Pool(X[tr_idx], y[tr_idx], feature_names=features)
        valid_pool = Pool(X[va_idx], y[va_idx], feature_names=features)

        model = CatBoostRegressor(**param)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=EARLY_STOPPING,
            use_best_model=True,
            verbose=False,
        )
        pred_residual = model.predict(X[va_idx])

        # Reconstruct TVT from the persistence anchor, same as your CV loop.
        y_true_tvt = train_df["target"].values[va_idx]
        last_tvt = train_df["last_known_tvt"].values[va_idx]
        # pred_tvt = last_tvt + pred_residual
        pred_tvt = pred_residual

        rmse_tvt = float(np.sqrt(np.mean((pred_tvt - y_true_tvt) ** 2)))
        fold_rmse_tvt.append(rmse_tvt)

        trial.report(np.mean(fold_rmse_tvt), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_rmse_tvt))


# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(
    study_name="cbt_TVT_tuning_20260729",
    storage="sqlite:///cbt_TVT_tuning_v2.db",
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, timeout=20 * 3600, n_jobs=1)  # NOTE: n_jobs>1 is unsafe on a single GPU (see below)

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best average rmse:", study.best_value)

best_score = study.best_value

study.trials_dataframe().to_csv(f"cbt_rogii_{best_score:.6f}.csv", index=False)

#### Check optuna results ###
s = optuna.load_study(study_name="cbt_TVT_tuning_20260729", storage="sqlite:///cbt_TVT_tuning_v2.db")
print(s.best_value, s.best_params)


###### Retrain a final model using best params, wrapped in a Trainer #####
# Unlike LightGBM, CatBoost's sklearn-style CatBoostRegressor IS the
# native-training API, so no param-name translation is needed here.
best_params_cb = study.best_params.copy()
best_params_cb.update({
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": NUM_BOOST_ROUND,
    "task_type": "GPU",
    "devices": "0",
    "verbose": False,
    "random_seed": SEED,
})

save_path = str(Path.cwd() / f"cb_final_{best_score:.6f}")
print(f"Models will be saved under: {save_path}")

# Trainer.fit() indexes folds with X.iloc[...], so it needs a DataFrame here,
# separate from the numpy `X` used above in the Optuna objective() loop
# (which relies on plain positional fancy-indexing X[tr_idx]).
X_df = train_df[features]

trainer = Trainer(
    estimator=CatBoostRegressor(**best_params_cb),
    task="regression",
    metric=root_mean_squared_error,
    cv=gkf,
    cv_args={"groups": g},
    use_early_stopping=True,
    verbose=True,
    save=True,
    save_path=save_path,
)

trainer.fit(
    X_df,
    y,
    fit_args={
        "early_stopping_rounds": EARLY_STOPPING,
        "use_best_model": True,
        "verbose": 250,
    },
)

print(f"Trainer overall RMSE: {trainer.overall_score:.6f}")
saved_files = list(Path(save_path).glob("*.pkl"))
if saved_files:
    for f in saved_files:
        print(f"Saved: {f.resolve()}")
else:
    print(f"WARNING: no .pkl found under {Path(save_path).resolve()} - save likely did not run or trainer.save was False.")
