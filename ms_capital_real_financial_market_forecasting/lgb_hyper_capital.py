"""
lgb_hyper_capital.py
Optuna hyperparameter tuning for the ms-capital LightGBM baseline.

Reads tr.csv (produced by your feature-engineering script), reuses the same
month-based train/valid split, and tunes LightGBM params to maximize the
competition's uncentered-cosine metric. Trials are persisted to a SQLite
study so tuning can be paused/resumed and inspected later.

Usage:
    python lgb_hyper_capital.py
"""

import time
import numpy as np
import polars as pl
import optuna
import lightgbm as lgb
import gc, time

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_PATH = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
#TR_CSV = "train.csv"
N_TRIALS = 5000
STUDY_NAME = "ms_capital_lgb_20260826"
STORAGE = "sqlite:///ms_capital_lgb_tuning.db"
GPU = True  # flip to True to use your OpenCL GPU backend (device="gpu")

VALID_MONTH_LO = 50   # train: month <= 50
VALID_MONTH_HI = 70   # valid: 50 < month <= 70


# def cos_uncenter(a, b):
#     return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def cosine_similarity_score(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    pred_centered = y_pred - y_pred.mean()
    true_centered = y_true - y_true.mean()
    
    cos_sim = (pred_centered * true_centered).sum() / (np.linalg.norm(pred_centered) + 1e-8) / (np.linalg.norm(true_centered) + 1e-8)
    return float(cos_sim)

# --------------------------------------------------------------------------
# Data (loaded once, reused across all trials)
# --------------------------------------------------------------------------
def load_data():
    tr = pl.read_csv(BASE_PATH+'\\processed_data\\train.csv')
    feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
    
    
    #tr_df = tr.filter(pl.col("month") <= VALID_MONTH_LO)
    tr_df = tr.filter(pl.col("sample_id") <= 904390)
    #va_df = tr.filter((pl.col("month") > VALID_MONTH_LO) & (pl.col("month") <= VALID_MONTH_HI))
    va_df = tr.filter((pl.col("sample_id") > 904390) & (pl.col("sample_id") <= 1257636))
    #print(va_df.select("sample_id").describe())

    X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    return X_tr, y_tr, X_va, y_va, feat_cols


X_TR, Y_TR, X_VA, Y_VA, FEAT_COLS = load_data()
print(f"train: {X_TR.shape}, valid: {X_VA.shape}, n_features={len(FEAT_COLS)}", flush=True)

NUM_BOOST_ROUND = 10000
EARLY_STOPPING = 200


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def objective(trial):
    param = dict(
        objective="regression",
        metric="rmse",
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 255),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 100, 10000),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 12),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        #max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
        #min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        max_depth=trial.suggest_int("max_depth", 2, 64),
        verbose=-1,
        #num_threads=16 ,
        seed=0,
    )
    if GPU:
        param.update(device="gpu", gpu_platform_id=0, gpu_device_id=0)

    # Built fresh each trial (like the ROGII script) so max_bin / min_data_in_leaf
    # can vary freely without hitting LightGBM's "Dataset already constructed" errors.
    dtrain = lgb.Dataset(X_TR, Y_TR, feature_name=FEAT_COLS)
    dvalid = lgb.Dataset(X_VA, Y_VA, feature_name=FEAT_COLS, reference=dtrain)

    model = lgb.train(
        param, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )

    p_va = model.predict(X_VA, num_iteration=model.best_iteration)
    v_cos = cosine_similarity_score(p_va, Y_VA)

    trial.set_user_attr("best_iteration", model.best_iteration)
    trial.set_user_attr("cos_similarity", v_cos)

    # Optuna minimizes; negate cosine so a higher cosine = lower (better) loss.
    return v_cos


# --------------------------------------------------------------------------
# Run study
# --------------------------------------------------------------------------
sampler = optuna.samplers.TPESampler(seed=0, multivariate=True)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE,
    load_if_exists=True,
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

t0 = time.time()
study.optimize(objective, timeout=0.5*3600, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\ntuning took {time.time() - t0:.1f}s", flush=True)

print(f"\nbest cos_similarity = {study.best_value:.4f}")
print("best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"best_iteration: {study.best_trial.user_attrs.get('best_iteration')}")

study.trials_dataframe().sort_values("value").to_csv("optuna_trials_20260826.csv", index=False)
print("\nall trials saved to optuna_trials.csv")




# --------------------------------------------------------------------------
# Create submission
# --------------------------------------------------------------------------
OUT_CSV = 'lgb_submission_140807_2.csv'
BASE_PATH = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
tr = pl.read_csv(BASE_PATH+'\\processed_data\\train.csv')
te_feats = pl.read_csv(BASE_PATH+'\\processed_data\\test.csv')

feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
print(f"n_features={len(feat_cols)}", flush=True)
# tr_df = tr.filter(pl.col("month") <= 50)
# va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
tr_df = tr.filter(pl.col("sample_id") <= 904390)
va_df = tr.filter((pl.col("sample_id") > 904390) & (pl.col("sample_id") <= 1257636))
del tr; gc.collect()

X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
y_tr = tr_df["target"].to_numpy().astype(np.float32)
X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)
del tr_df, va_df; gc.collect()


def cosine_similarity_score(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    pred_centered = y_pred - y_pred.mean()
    true_centered = y_true - y_true.mean()
    
    cos_sim = (pred_centered * true_centered).sum() / (np.linalg.norm(pred_centered) + 1e-8) / (np.linalg.norm(true_centered) + 1e-8)
    return float(cos_sim)

# --- NEW: custom feval so LightGBM early-stops/monitors on cosine similarity, not RMSE ---
def cos_sim_feval(preds, train_data):
    y_true = train_data.get_label()
    val = cosine_similarity_score(preds, y_true)
    return "cos_sim", val, True  # True = higher is better

print(f"\ntrain X: {X_tr.shape}, valid X: {X_va.shape}", flush=True)
params = dict( # 0.136795
    objective="regression",   # L2 (MSE) loss - RMSE 优化同样目标 
    # metric="rmse",          # REMOVED: no longer the metric LightGBM reports/early-stops on
    metric="None",             # tells LightGBM not to compute its built-in metric, only feval
    learning_rate=0.00550010656425676, 
    num_leaves=225, 
    min_data_in_leaf=940,
    feature_fraction=0.5913798513263131, 
    bagging_fraction=0.8456385300816562, 
    bagging_freq=1,
    lambda_l1=0.03128170303963041,
    lambda_l2=1.679513133436344, 
    #max_bin=255, 
    #min_gain_to_split=0.00022526657860905087,
    max_depth=33,
    verbose=-1, 
    #num_threads=16, 
    seed=0 
    )
dtr = lgb.Dataset(X_tr, y_tr)
dva = lgb.Dataset(X_va, y_va, reference=dtr)
t0 = time.time()
model = lgb.train(params, dtr, num_boost_round=10000,
                  valid_sets=[dtr, dva],
                  valid_names=["train", "valid"],
                  feval=cos_sim_feval,
                  callbacks=[lgb.early_stopping(200, first_metric_only=True),
                             lgb.log_evaluation(period=100)])
print(f"train took {time.time()-t0:.1f}s, best_iter={model.best_iteration}", flush=True)

p_va = model.predict(X_va, num_iteration=model.best_iteration)
p_tr = model.predict(X_tr, num_iteration=model.best_iteration)
v_cos = cosine_similarity_score(p_va, y_va)
t_cos = cosine_similarity_score(p_tr, y_tr)
print(f"\n>>> train_cos = {t_cos:.6f}", flush=True)
print(f">>> valid_cos = {v_cos:.6f}", flush=True)

# def cos_uncenter(a, b):
#     return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


p_va = model.predict(X_va, num_iteration=model.best_iteration)
v_cos = cosine_similarity_score(p_va, y_va)
print(f"\n>>> valid_cos = {v_cos:.6f}", flush=True)


####### Feature Importance #######
fi_gain = model.feature_importance(importance_type="gain")
fi_split = model.feature_importance(importance_type="split")
fi_df = pl.DataFrame({
    "feature": feat_cols,
    "gain": fi_gain,
    "split": fi_split,
}).sort("gain", descending=True)
fi_df.write_csv("feature_importance.csv")
print(f"\n feature importance saved: feature_importance_lgb.csv", flush=True)
print(fi_df.head(20), flush=True)


X_te = te_feats.select(feat_cols).to_numpy().astype(np.float32)
ids_te = te_feats["sample_id"].to_numpy()
p_te = model.predict(X_te, num_iteration=model.best_iteration)
pl.DataFrame({"sample_id": pl.Series(ids_te, dtype=pl.Int32),
              "prediction": pl.Series(p_te, dtype=pl.Float64)}).sort("sample_id").write_csv(OUT_CSV)
print(f"submission saved: {OUT_CSV}", flush=True)


