# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:31:53 2026

@author: azz
"""

from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import optuna
from optuna.samplers import TPESampler
from koolbox import Trainer
import pickle

SEED = 42 

DATA_ROOT = Path(r"H:\kaggle\rogii-wellbore-geology-prediction")
DF = "train_df.csv"

train_df = pd.read_csv(DATA_ROOT / DF, dtype={"well": "string"})


features = [c for c in train_df.columns if c not in {'well','id','target'}]

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
        "objective": "regression",
        #"n_estimators": trial.suggest_categorical("n_estimators", [500, 1000, 1500, 2000])
        "n_estimators": 5000,
        "metric": "rmse",  
        #"boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "feature_fraction": trial.suggest_categorical("feature_fraction", [0.8, 0.85, 0.9, 0.95]),
        "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.8, 0.85, 0.9, 0.95]),
        "bagging_freq": trial.suggest_int("bagging_freq", 5, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 12, 256),
        "max_depth": trial.suggest_int("max_depth", 2, 64),  # -1 means no limit
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        #"min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        "device_type": "gpu",  # Enable GPU support
        "verbosity": -1,
        "seed" : SEED

    }

    fold_rmse_tvt = []
    for fold, (tr_idx, va_idx) in enumerate(cv_folds):
        assert tr_idx.max() < X.shape[0] and tr_idx.min() >= 0, \
           f"corrupt tr_idx at fold {fold}: max={tr_idx.max()}, min={tr_idx.min()}, X rows={X.shape[0]}"
        assert va_idx.max() < X.shape[0] and va_idx.min() >= 0, \
           f"corrupt va_idx at fold {fold}: max={va_idx.max()}, min={va_idx.min()}"
        dtrain = lgb.Dataset(X[tr_idx], y[tr_idx], feature_name=features)
        dvalid = lgb.Dataset(X[va_idx], y[va_idx], feature_name=features, reference=dtrain)

        model = lgb.train(
            param, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(10)],
        )
        pred_residual = model.predict(X[va_idx], num_iteration=model.best_iteration)

        # Reconstruct TVT from the persistence anchor, same as your CV loop.
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
 

# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(
    study_name="lgb_TVT_tuning_20260728",
    storage="sqlite:///lgb_TVT_tuning.db" ,
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, timeout=22*3600, n_jobs=3) # n hour

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
s = optuna.load_study(study_name="lgb_TVT_tuning_20260728", storage="sqlite:///lgb_TVT_tuning.db")
print(s.best_value, s.best_params)
#s.trials_dataframe().tail(10) 



###### Retrain a final model using best params, wrapped in a Trainer #####
# Native-API param names (from the Optuna search space) -> sklearn-API
# (LGBMRegressor) equivalents. Without this mapping these values would
# silently be ignored by LGBMRegressor.
_native_to_sklearn = {
    "feature_fraction": "colsample_bytree",
    "bagging_fraction": "subsample",
    "bagging_freq": "subsample_freq",
    "min_data_in_leaf": "min_child_samples",
    "lambda_l1": "reg_alpha",
    "lambda_l2": "reg_lambda",
}

best_params_sklearn = {
    _native_to_sklearn.get(k, k): v for k, v in study.best_params.items()
}
best_params_sklearn.update({
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": NUM_BOOST_ROUND,
    "device_type": "gpu",
    "verbosity": -1,
    "seed": SEED,
    "n_jobs": -1,
})

save_path = str(Path.cwd() / f"lgb_final_{best_score:.6f}")
print(f"Models will be saved under: {save_path}")

# Trainer.fit() indexes folds with X.iloc[...], so it needs a DataFrame here,
# separate from the numpy `X` used above in the Optuna objective() loop
# (which relies on plain positional fancy-indexing X[tr_idx]).
X_df = train_df[features]

trainer = Trainer(
    estimator=LGBMRegressor(**best_params_sklearn),
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
        "eval_metric": "rmse",
        "callbacks": [
            log_evaluation(period=250),
            early_stopping(stopping_rounds=EARLY_STOPPING),
        ],
    },
)

print(f"Trainer overall RMSE: {trainer.overall_score:.6f}")
saved_files = list(Path(save_path).glob("*.pkl"))
if saved_files:
    for f in saved_files:
        print(f"Saved: {f.resolve()}")
else:
    print(f"WARNING: no .pkl found under {Path(save_path).resolve()} - save likely did not run or trainer.save was False.")