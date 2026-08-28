"""
catboost_hyper_capital.py
Optuna hyperparameter tuning for the ms-capital CatBoost baseline.

Mirrors lgb_hyper_capital.py: reads train.csv (produced by your feature-
engineering script), reuses the same sample_id-based train/valid split, and
tunes CatBoost params to maximize the competition's centered-cosine metric
(cosine_similarity_score below — swap for cos_uncenter if your competition's
actual metric is uncentered, per your earlier check). Trials are persisted
to a SQLite study so tuning can be paused/resumed and inspected later.

Usage:
    python catboost_hyper_capital.py
"""

import time
import gc
import numpy as np
import polars as pl
import optuna
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_PATH = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
N_TRIALS = 5000
STUDY_NAME = "ms_capital_catboost_20260827"
STORAGE = "sqlite:///ms_capital_catboost_tuning.db"
GPU = True  # flip to True to train on GPU (task_type="GPU")

VALID_MONTH_LO = 50   # kept for reference; actual split below uses sample_id
VALID_MONTH_HI = 70


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
# Custom CatBoost eval metric — CatBoost expects a class with this interface,
# not a plain function like LightGBM's feval.
# --------------------------------------------------------------------------
class CosSimMetric:
    """Custom CatBoost eval metric wrapping cosine_similarity_score.

    CatBoost calls `evaluate` every eval period with raw model outputs
    (`approxes`) and true labels (`target`); `is_max_optimal` tells CatBoost
    whether higher is better (so early stopping / best-iteration selection
    know which direction to optimize in), and `get_final_error` is called
    once at the end to reduce the per-batch errors into the final score
    (trivial here since we compute over the whole eval set in one call).
    """

    def is_max_optimal(self):
        return True  # higher cosine similarity is better

    def evaluate(self, approxes, target, weight):
        # approxes is a list with one array per model output (1 for regression)
        preds = np.asarray(approxes[0])
        y_true = np.asarray(target)
        score = cosine_similarity_score(preds, y_true)
        return score, 1.0  # (score, weight_sum) — weight_sum=1 since already reduced

    def get_final_error(self, error, weight):
        return error


# --------------------------------------------------------------------------
# Data (loaded once, reused across all trials)
# --------------------------------------------------------------------------
def load_data():
    tr = pl.read_csv(BASE_PATH + '\\processed_data\\train.csv')
    feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]

    tr_df = tr.filter(pl.col("sample_id") <= 904390)
    va_df = tr.filter((pl.col("sample_id") > 904390) & (pl.col("sample_id") <= 1257636))

    X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    return X_tr, y_tr, X_va, y_va, feat_cols


X_TR, Y_TR, X_VA, Y_VA, FEAT_COLS = load_data()
print(f"train: {X_TR.shape}, valid: {X_VA.shape}, n_features={len(FEAT_COLS)}", flush=True)

# CatBoost Pools built once and reused across trials (analogous to lgb.Dataset)
TRAIN_POOL = Pool(X_TR, Y_TR, feature_names=FEAT_COLS)
VALID_POOL = Pool(X_VA, Y_VA, feature_names=FEAT_COLS)

NUM_BOOST_ROUND = 10000
EARLY_STOPPING = 200


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def objective(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
    params = dict(
        loss_function="RMSE",
        eval_metric=CosSimMetric(),
        iterations=NUM_BOOST_ROUND,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        depth=trial.suggest_int("depth", 4, 8),  # clipped for GPU safety — verify your version's actual cap
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        #border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
        grow_policy=grow_policy,
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )
    if grow_policy != "SymmetricTree":
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 1000)
    if GPU:
        params.update(task_type="GPU", devices="0")
    else:
        params.update(task_type="CPU")

    try:
        model = CatBoostRegressor(**params)
        model.fit(TRAIN_POOL, eval_set=VALID_POOL, use_best_model=True,
                   early_stopping_rounds=EARLY_STOPPING, verbose=False)
        p_va = model.predict(X_VA)
        v_cos = cosine_similarity_score(p_va, Y_VA)
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        trial.set_user_attr("cos_similarity", v_cos)
        return v_cos
    except Exception as e:
        print(f"trial failed: {e}", flush=True)
        return -1.0

    model = CatBoostRegressor(**params)
    model.fit(
        TRAIN_POOL,
        eval_set=VALID_POOL,
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING,
        verbose=False,
    )

    p_va = model.predict(X_VA)
    v_cos = cosine_similarity_score(p_va, Y_VA)

    trial.set_user_attr("best_iteration", model.get_best_iteration())
    trial.set_user_attr("cos_similarity", v_cos)

    # Optuna direction="maximize" below, so return the raw cosine value.
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
study.optimize(objective, timeout= 1*3600, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\ntuning took {time.time() - t0:.1f}s", flush=True)

print(f"\nbest cos_similarity = {study.best_value:.6f}")
print("best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"best_iteration: {study.best_trial.user_attrs.get('best_iteration')}")

study.trials_dataframe().sort_values("value").to_csv("optuna_trials_catboost_20260827.csv", index=False)
print("\nall trials saved to optuna_trials_catboost_20260827.csv")


# --------------------------------------------------------------------------
# Create submission (using best Optuna params)
# --------------------------------------------------------------------------
OUT_CSV = 'catboost_submission_140022.csv'
BASE_PATH = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
tr = pl.read_csv(BASE_PATH + '\\processed_data\\train.csv')
te_feats = pl.read_csv(BASE_PATH + '\\processed_data\\test.csv')

feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
print(f"n_features={len(feat_cols)}", flush=True)
tr_df = tr.filter(pl.col("sample_id") <= 904390)
va_df = tr.filter((pl.col("sample_id") > 904390) & (pl.col("sample_id") <= 1257636))
del tr
gc.collect()

X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
y_tr = tr_df["target"].to_numpy().astype(np.float32)
X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)
del tr_df, va_df
gc.collect()

print(f"\ntrain X: {X_tr.shape}, valid X: {X_va.shape}", flush=True)

final_params = dict(
    loss_function="RMSE",
    eval_metric=CosSimMetric(),
    iterations=10000,
    random_seed=0,
    verbose=False,
    allow_writing_files=False,
    **study.best_params,   # learning_rate, depth, l2_leaf_reg, random_strength,
                           # bagging_temperature, border_count, min_data_in_leaf
)
if GPU:
    final_params.update(task_type="GPU", devices="0")
else:
    final_params.update(task_type="CPU")

train_pool = Pool(X_tr, y_tr, feature_names=feat_cols)
valid_pool = Pool(X_va, y_va, feature_names=feat_cols)

t0 = time.time()
model = CatBoostRegressor(**final_params)
model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
    early_stopping_rounds=200,
    verbose=100,   # prints train/valid metric every 100 rounds, like log_evaluation(period=100)
)
print(f"train took {time.time()-t0:.1f}s, best_iter={model.get_best_iteration()}", flush=True)

p_va = model.predict(X_va)
p_tr = model.predict(X_tr)
v_cos = cosine_similarity_score(p_va, y_va)
t_cos = cosine_similarity_score(p_tr, y_tr)
print(f"\n>>> train_cos = {t_cos:.6f}", flush=True)
print(f">>> valid_cos = {v_cos:.6f}", flush=True)

####### Feature Importance #######
fi_values = model.get_feature_importance(train_pool, type="FeatureImportance")
fi_df = pl.DataFrame({
    "feature": feat_cols,
    "importance": fi_values,
}).sort("importance", descending=True)
fi_df.write_csv("feature_importance_catboost.csv")
print(f"\n feature importance saved: feature_importance_catboost.csv", flush=True)
print(fi_df.head(20), flush=True)

X_te = te_feats.select(feat_cols).to_numpy().astype(np.float32)
ids_te = te_feats["sample_id"].to_numpy()
p_te = model.predict(X_te)
pl.DataFrame({"sample_id": pl.Series(ids_te, dtype=pl.Int32),
              "prediction": pl.Series(p_te, dtype=pl.Float64)}).sort("sample_id").write_csv(OUT_CSV)
print(f"submission saved: {OUT_CSV}", flush=True)
