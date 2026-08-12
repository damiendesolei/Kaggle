import os
import gc, time
import numpy as np
import polars as pl
import lightgbm as lgb

DATA = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
OUT_CSV = "submission_baseline.csv"

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def market_feats(split):
    """~40 market features. Lazy scan + 单次 collect."""
    lf = pl.scan_ipc(f"{DATA}/{split}/market.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_sp"),
        ((pl.col("ask_volume_1") - pl.col("bid_volume_1"))
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).alias("_imb"),
        (pl.col("ask_volume_1") + pl.col("bid_volume_1")).alias("_depth"),
    ])
    lf = lf.with_columns([
        pl.col("_mid").diff().over("sample_id").fill_null(0.0).alias("_dmid"),
        (pl.col("ask_volume_1").diff().over("sample_id").fill_null(0.0)
         - pl.col("bid_volume_1").diff().over("sample_id").fill_null(0.0)).alias("_ofi"),
    ])
    # 用 filter-in-agg 方式一次 group_by 出所有 window stats + 全序列
    exprs = [
        # 全序列 12
        pl.col("_mid").last().alias("m_mid_last"),
        pl.col("_mid").mean().alias("m_mid_mean"),
        pl.col("_mid").std().alias("m_mid_std"),
        (pl.col("_mid").max() - pl.col("_mid").min()).alias("m_mid_range"),
        pl.col("_sp").last().alias("m_sp_last"),
        pl.col("_sp").mean().alias("m_sp_mean"),
        pl.col("_imb").last().alias("m_imb_last"),
        pl.col("_imb").mean().alias("m_imb_mean"),
        pl.col("_imb").std().alias("m_imb_std"),
        pl.col("_depth").mean().alias("m_depth_mean"),
        (pl.col("_dmid") ** 2).sum().sqrt().alias("m_rv"),
        pl.col("_ofi").sum().alias("m_ofi_sum"),
    ]
    # 3 短窗口, 每个 7 feats (agg 时用 filter 表达式)
    for w in [15, 60, 180]:
        cond = pl.col("seconds_before_predict") <= w
        exprs += [
            pl.col("_mid").filter(cond).mean().alias(f"m_mid_mean_{w}"),
            pl.col("_mid").filter(cond).std().alias(f"m_mid_std_{w}"),
            pl.col("_sp").filter(cond).mean().alias(f"m_sp_mean_{w}"),
            pl.col("_imb").filter(cond).mean().alias(f"m_imb_mean_{w}"),
            ((pl.col("_dmid").filter(cond)) ** 2).sum().sqrt().alias(f"m_rv_{w}"),
            pl.col("_ofi").filter(cond).sum().alias(f"m_ofi_sum_{w}"),
            pl.col("transaction_volume").filter(cond).sum().alias(f"m_txv_sum_{w}"),
        ]
    # EWM 2 τ × 3 signals = 6
    for tau in [30, 120]:
        w_expr = pl.col("seconds_before_predict").mul(-1.0 / tau).exp()
        exprs += [
            ((w_expr * pl.col("_mid")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_mid_ewm_{tau}"),
            ((w_expr * pl.col("_imb")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_imb_ewm_{tau}"),
            ((w_expr * pl.col("_ofi")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_ofi_ewm_{tau}"),
        ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)


def tx_feats(split):
    """~25 tx features. Lazy + single agg."""
    lf = pl.scan_ipc(f"{DATA}/{split}/transaction.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        pl.col("volume").log1p().alias("_lv"),
    ])
    lf = lf.with_columns([
        (pl.col("_sgn") * pl.col("volume")).alias("_sv"),
        (pl.col("_sgn") * pl.col("price") * pl.col("volume")).alias("_sd"),
    ])
    exprs = [
        pl.col("volume").sum().alias("t_vol_sum"),
        pl.col("_sv").sum().alias("t_sv_sum"),
        pl.col("_sd").sum().alias("t_sd_sum"),
        pl.col("_lv").mean().alias("t_lv_mean"),
        pl.col("price").std().alias("t_px_std"),
        pl.col("price").last().alias("t_px_last"),
        (pl.col("_sgn") > 0).mean().alias("t_buy_ratio"),
    ]
    for w in [15, 45, 120]:
        cond = pl.col("seconds_before_predict") <= w
        exprs += [
            pl.col("volume").filter(cond).sum().alias(f"t_vol_{w}"),
            pl.col("_sv").filter(cond).sum().alias(f"t_sv_{w}"),
            pl.col("_sd").filter(cond).sum().alias(f"t_sd_{w}"),
            pl.col("_lv").filter(cond).mean().alias(f"t_lv_mean_{w}"),
            (pl.col("_sgn").filter(cond) > 0).mean().alias(f"t_buy_ratio_{w}"),
            pl.col("_sgn").filter(cond).count().alias(f"t_n_{w}"),
        ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)


def ord_feats(split):
    """~19 order features. Lazy + single agg."""
    lf = pl.scan_ipc(f"{DATA}/{split}/order.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        (pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_act"),
    ])
    lf = lf.with_columns([
        (pl.col("_sgn") * pl.col("volume")).alias("_sv"),
        (pl.col("_act") * pl.col("volume")).alias("_av"),
    ])
    exprs = [
        pl.col("volume").sum().alias("o_vol_sum"),
        pl.col("_sv").sum().alias("o_sv_sum"),
        pl.col("_av").sum().alias("o_av_sum"),
        (pl.col("_sgn") > 0).mean().alias("o_buy_ratio"),
        (pl.col("_act") > 0).mean().alias("o_add_ratio"),
        pl.col("price").std().alias("o_px_std"),
    ]
    for w in [15, 45, 120]:
        cond = pl.col("seconds_before_predict") <= w
        exprs += [
            pl.col("_sv").filter(cond).sum().alias(f"o_sv_{w}"),
            pl.col("_av").filter(cond).sum().alias(f"o_av_{w}"),
            (pl.col("_sgn").filter(cond) > 0).mean().alias(f"o_buy_ratio_{w}"),
            pl.col("_sgn").filter(cond).count().alias(f"o_n_{w}"),
        ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)


def cross_feats(df):
    """~8 cross-source features."""
    return df.with_columns([
        (pl.col("m_sp_mean") * pl.col("m_imb_mean")).alias("x_sp_imb"),
        (pl.col("t_sv_sum") / (pl.col("t_vol_sum") + 1.0)).alias("x_t_signed_ratio"),
        (pl.col("o_sv_sum") / (pl.col("o_vol_sum") + 1.0)).alias("x_o_signed_ratio"),
        (pl.col("t_sd_sum") / (pl.col("t_vol_sum") + 1.0) - pl.col("m_mid_last")).alias("x_vwap_vs_mid"),
        (pl.col("m_ofi_ewm_30") - pl.col("m_ofi_ewm_120")).alias("x_ofi_ewm_short_long"),
        (pl.col("m_imb_ewm_30") - pl.col("m_imb_ewm_120")).alias("x_imb_ewm_short_long"),
        (pl.col("t_sv_15") / (pl.col("t_vol_15") + 1.0)).alias("x_t_signed_ratio_15"),
        (pl.col("m_rv_15") / (pl.col("m_rv") + 1e-8)).alias("x_rv_15_over_full"),
    ])


def build_features(split):
    print(f"\n=== {split} ===", flush=True)
    t0 = time.time()
    mf = market_feats(split); gc.collect()
    print(f"  market: {mf.shape}  ({time.time()-t0:.1f}s)", flush=True)
    tf = tx_feats(split); gc.collect()
    print(f"  tx: {tf.shape}", flush=True)
    of = ord_feats(split); gc.collect()
    print(f"  order: {of.shape}", flush=True)
    df = mf.join(tf, on="sample_id", how="left").join(of, on="sample_id", how="left")
    df = cross_feats(df)
    print(f"  final: {df.shape}  total {time.time()-t0:.1f}s", flush=True)
    return df


def main():
    tr_feats = build_features("train")
    te_feats = build_features("test")

    label = pl.read_ipc(f"{DATA}/train/label.feather", memory_map=False)
    tr = tr_feats.join(label, on="sample_id", how="inner")
    del tr_feats; gc.collect()
    print(f"\ntrain w/ label: {tr.shape}", flush=True)

    feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
    print(f"n_features={len(feat_cols)}", flush=True)
    tr_df = tr.filter(pl.col("month") <= 50)
    va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
    del tr; gc.collect()

    X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    del tr_df, va_df; gc.collect()

    print(f"\ntrain X: {X_tr.shape}, valid X: {X_va.shape}", flush=True)
    params = dict(
        objective="regression",   # L2 (MSE) loss - RMSE 优化同样目标
        metric="rmse",
        learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=16, seed=0)
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(period=100)])
    print(f"train took {time.time()-t0:.1f}s, best_iter={model.best_iteration}", flush=True)

    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    v_cos = cos_uncenter(p_va, y_va)
    print(f"\n>>> valid_cos = {v_cos:.6f}", flush=True)

    X_te = te_feats.select(feat_cols).to_numpy().astype(np.float32)
    ids_te = te_feats["sample_id"].to_numpy()
    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    pl.DataFrame({"sample_id": pl.Series(ids_te, dtype=pl.Int32),
                  "prediction": pl.Series(p_te, dtype=pl.Float64)}).sort("sample_id").write_csv(OUT_CSV)
    print(f"submission saved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()

