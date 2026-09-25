#!/usr/bin/env python3
"""
realmlp_capital_tune.py

Optuna hyperparameter tuning for the RealMLP_RQ model used in
MS Capital - Real Financial Market Forecasting.

Reuses the model/data pipeline from realmlp_capital.py but:
  - loads and preprocesses the data ONCE (independent of hyperparameters)
  - wraps model construction + training in an `objective(trial)` function
  - trains each trial for a small number of epochs on a (optionally
    subsampled) training set, reporting validation cosine similarity
    after every epoch so Optuna can prune unpromising trials early
  - uses a SQLite-backed study (TPE sampler) so tuning can be resumed
    across runs, mirroring the LightGBM tuning setup used elsewhere
  - writes the best hyperparameters to a JSON file at the end

Usage:
    python realmlp_capital_tune.py --n-trials 200
    python realmlp_capital_tune.py --n-trials 50 --tune-epochs 4 --tune-train-size 300000

Resuming is automatic: re-running with the same --study-name/--storage
will continue adding trials to the same study.
"""

# ===================================================================
# Imports & global config
# ===================================================================
import argparse
import json
import math
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

BASE_PATH = r"H:\kaggle\ms-capital-real-financial-market-forecasting"
onehotmax = 10
target_col = "target"
eval_bs = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===================================================================
# Preprocessing utilities (unchanged from realmlp_capital.py)
# ===================================================================
def filter_high_correlation(df, target_col, corr_threshold=0.95, method="pearson"):
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    target_corr = {}
    for col in feature_cols:
        try:
            corr = X[col].corr(y, method=method)
            target_corr[col] = abs(corr) if not np.isnan(corr) else -1
        except Exception:
            target_corr[col] = -1

    corr_matrix = X.corr(method=method)

    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= corr_threshold:
                high_corr_pairs.append((cols[i], cols[j], abs(corr_val)))
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    drop_cols = set()
    for col1, col2, _ in high_corr_pairs:
        if col1 in drop_cols or col2 in drop_cols:
            continue
        if target_corr[col1] >= target_corr[col2]:
            drop_cols.add(col2)
        else:
            drop_cols.add(col1)

    for col in feature_cols:
        if col not in drop_cols and target_corr[col] < 0.0001:
            drop_cols.add(col)

    return list(drop_cols)


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != "object":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduced)")
    return df


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (np.max(X, axis=-2)[idxs] - np.min(X, axis=-2)[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)


# ===================================================================
# Model building blocks (unchanged, plus tunable dims where noted)
# ===================================================================
class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim=8, onehotmax_=onehotmax):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features, self.embed_features, self.embed_dims = [], [], []
        self.embed_offsets = []

        for i, dim in enumerate(cat_dims):
            if dim <= onehotmax_:
                self.onehot_features.append(i)
            else:
                self.embed_features.append(i)
                self.embed_dims.append(dim)

        if self.embed_features:
            total_vocab = sum(self.embed_dims) * n_ens
            self.combined_emb = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
            offset = 0
            for dim in self.embed_dims:
                self.embed_offsets.append(offset)
                offset += dim
            self.per_ens_offset = sum(self.embed_dims)

    def forward(self, x):
        batch_size, n_ens, n_cat = x.shape
        features = []

        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_onehot_dim = sum(onehot_dims)
            onehot_encoded = torch.zeros(batch_size, n_ens, total_onehot_dim, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx + 1].long()
                onehot_encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(onehot_encoded)

        if self.embed_features:
            embed_x = x[:, :, self.embed_features].long()
            ens_offset = torch.arange(n_ens, device=x.device) * self.per_ens_offset
            feat_offset = torch.tensor(self.embed_offsets, device=x.device)
            indices = embed_x + feat_offset.unsqueeze(0).unsqueeze(0) + ens_offset.unsqueeze(0).unsqueeze(-1)
            embedded = self.combined_emb(indices)
            embedded = embedded.view(batch_size, n_ens, -1)
            features.append(embedded)

        # Handle the case where there are no categorical features.
        if not features:
            return torch.empty(
                batch_size, n_ens, 0,
                device=x.device,
                dtype=torch.float32,
            )

        return torch.cat(features, dim=2)


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16,
                 out_dim: int = 4, freq_scale: float = 0.1):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)

    def forward(self, x):
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)
        periodic = torch.cos(2 * np.pi * (x_expanded * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = torch.einsum("b n f h, n f h d -> b n f d", periodic, self.w2)
        transformed = self.act(transformed + self.b2.unsqueeze(0))
        result = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return result.view(batch_size, self.n_ens, -1)


class NTPLinear(nn.Module):
    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        assert x.ndim == 3
        x = torch.einsum("b n i, n i o -> b n o", x, self.weight) / np.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class RQKMeansEncoder:
    def __init__(self, n_layers=4, codebook_size=5):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.codebooks = []

    def fit(self, y):
        residuals = y.copy().reshape(-1, 1)
        for _ in range(self.n_layers):
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(residuals)
            self.codebooks.append(kmeans)
            codes = kmeans.predict(residuals)
            centroids = kmeans.cluster_centers_[codes].reshape(-1, 1)
            residuals = residuals - centroids
        return self

    def encode(self, y):
        residuals = y.copy().reshape(-1, 1)
        all_codes = []
        for kmeans in self.codebooks:
            codes = kmeans.predict(residuals)
            all_codes.append(codes)
            centroids = kmeans.cluster_centers_[codes].reshape(-1, 1)
            residuals = residuals - centroids
        return np.stack(all_codes, axis=1)


class RealMLP_RQ(nn.Module):
    """Same architecture as realmlp_capital.py, with the previously-hardcoded
    PBLD embedding dims, hidden-layer widths, and dropout exposed as
    constructor args so Optuna can tune them."""

    def __init__(self, output_dim=1, cat_dims=(), n_numerical=None,
                 n_ens=8, embed_dim=4, n_rq_layers=4, rq_vocab_size=5,
                 pbld_hidden_dim=24, pbld_out_dim=3, pbld_freq_scale=1.0,
                 hidden_dims=(512, 512, 128), dropout=0.01):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.embed_dim = embed_dim
        self.n_rq_layers = n_rq_layers
        self.rq_vocab_size = rq_vocab_size

        self.cate = CategoricalFeatureLayer(n_ens=n_ens, cat_dims=list(cat_dims), embed_dim=embed_dim)
        self.num_embed = PBLDEmbedding(
            n_features=n_numerical, hidden_dim=pbld_hidden_dim,
            out_dim=pbld_out_dim, freq_scale=pbld_freq_scale, n_ens=n_ens,
        )
        num_emb_dim = n_numerical * pbld_out_dim
        cat_emb_dim = sum([c if c <= onehotmax else embed_dim for c in cat_dims])
        total_dim = int(num_emb_dim + cat_emb_dim)

        layers = [nn.LayerNorm(total_dim), ScalingLayer(n_ens=n_ens, n_features=total_dim)]
        in_dim = total_dim
        for h in hidden_dims:
            layers += [NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=h),
                       act(), nn.Dropout(dropout)]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        self.code_heads = nn.ModuleList([
            NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=rq_vocab_size)
            for _ in range(n_rq_layers)
        ])
        self.reg_head = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=1)
        self.register_buffer("feature_mask", self._create_mask(total_dim))

    def _create_mask(self, n_features):
        mask = torch.ones(self.n_ens, n_features, dtype=torch.bool)
        step = max(self.n_ens // 2, 1)
        for i in range(self.n_ens):
            mask[i, i::step] = False
        return mask

    def forward(self, x_num, x_cat, return_codes=False):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        mask_expanded = self.feature_mask.unsqueeze(0).expand(combined.shape[0], -1, -1)
        combined = combined * mask_expanded.float()
        features = self.shared(combined)
        code_logits = [head(features) for head in self.code_heads]
        reg_pred = self.reg_head(features)
        if return_codes:
            return code_logits, reg_pred
        return reg_pred.mean(dim=1)


# ===================================================================
# Training utilities (unchanged)
# ===================================================================
def flat_anneal(init_value, progress, flat_ratio=0.5):
    if progress < flat_ratio:
        return init_value
    decay_progress = (progress - flat_ratio) / (1 - flat_ratio)
    return init_value * (1 - decay_progress)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_state = {name: p.data.clone().detach()
                           for name, p in model.named_parameters() if p.requires_grad}

    def update(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ema_state[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self):
        original_state = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                original_state[name] = p.data.clone().detach()
                p.data.copy_(self.ema_state[name])
        return original_state

    def restore(self, original_state):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in original_state:
                p.data.copy_(original_state[name])


def get_parameter_groups(model):
    scale_p, pbld_p, first_linear_p, other_w_p, bias_p = [], [], [], [], []
    first_linear_weight_id = None
    for name, param in model.named_parameters():
        if "shared.0.weight" in name:  # exact original realmlp_capital.py behavior
            first_linear_weight_id = id(param)
            break
    for name, param in model.named_parameters():
        if "scale" in name:
            scale_p.append(param)
        elif "num_embed" in name:
            pbld_p.append(param)
        elif first_linear_weight_id is not None and id(param) == first_linear_weight_id:
            first_linear_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    return scale_p, pbld_p, first_linear_p, other_w_p, bias_p


def cosine_similarity_score(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    pred_centered = y_pred - y_pred.mean()
    true_centered = y_true - y_true.mean()
    return float((pred_centered * true_centered).sum()
                  / (np.linalg.norm(pred_centered) + 1e-8)
                  / (np.linalg.norm(true_centered) + 1e-8))


def evaluate_model(model, x_num, x_cat, y_true, batch_size=eval_bs):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            pred = model(x_num[i:i + batch_size], x_cat[i:i + batch_size],
                         return_codes=False).mean(dim=1).squeeze()
            all_preds.append(pred.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    return cosine_similarity_score(all_preds, y_true_np), all_preds


def compute_loss_with_rq(y_pred, y_true, code_logits, y_codes, lambda_cos=0.01, lambda_rq=0.1):
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(-1)
    batch_size, n_ens = y_pred.shape
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_ens)
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true_expanded.reshape(-1)

    abs_true = torch.abs(y_true_flat)
    sample_weights = torch.where(abs_true > 0.001, 0.5, 1.0)
    mse_loss = (sample_weights * (y_pred_flat - y_true_flat) ** 2).mean()

    pred_centered = y_pred_flat - y_pred_flat.mean()
    true_centered = y_true_flat - y_true_flat.mean()
    cos_sim = (pred_centered * true_centered).sum() / (pred_centered.norm() + 1e-8) / (true_centered.norm() + 1e-8)
    cos_loss = 1 - cos_sim

    rq_loss = 0
    y_codes_expanded = y_codes.unsqueeze(1).expand(-1, n_ens, -1)
    for layer, logits in enumerate(code_logits):
        labels = y_codes_expanded[:, :, layer].reshape(-1)
        logits_flat = logits.reshape(-1, logits.size(-1))
        rq_loss += F.cross_entropy(logits_flat, labels, reduction="mean")
    rq_loss = rq_loss / len(code_logits)

    total_loss = mse_loss + lambda_cos * cos_loss + lambda_rq * rq_loss
    return total_loss, cos_sim, mse_loss, rq_loss


# ===================================================================
# Data loading (runs once, independent of hyperparameters)
# ===================================================================
DATA = {}


def load_data():
    """Reproduce the original realmlp_capital.py preprocessing and split."""
    set_seed(42)

    train = pd.read_csv(BASE_PATH + "\\processed_data\\train.csv").sort_values("sample_id")
    test = pd.read_csv(BASE_PATH + "\\processed_data\\test.csv").sort_values("sample_id")

    # Feature elimination disabled: keep all source-data features.
    # original_features = [c for c in train.columns if c not in ["sample_id", target_col]]
    # 
    # # Restore original feature elimination.
    # correlation_drop = filter_high_correlation(
    #     train, target_col, corr_threshold=0.9, method="pearson"
    # )
    # constant_drop = [c for c in train.columns if train[c].nunique() == 1]
    # drop_set = set(correlation_drop) | set(constant_drop)
    # drop = [c for c in train.columns if c in drop_set]
    # 
    # print("\n" + "=" * 70)
    # print("ORIGINAL FEATURE ELIMINATION")
    # print("=" * 70)
    # print(f"Original feature count  : {len(original_features)}")
    # print(f"Eliminated feature count: {len(drop)}")
    # print(f"Remaining feature count : {len(original_features) - len(drop)}")
    # if drop:
    #     print("\nEliminated features:")
    #     for i, col in enumerate(drop, 1):
    #         reasons = []
    #         if col in correlation_drop:
    #             reasons.append("correlation / low target correlation")
    #         if col in constant_drop:
    #             reasons.append("constant")
    #         print(f"  {i:>3}. {col}  [{', '.join(reasons)}]")
    # print("=" * 70 + "\n")
    # 
    # train.drop(drop, axis=1, inplace=True)
    # test.drop(drop, axis=1, inplace=True)

    # Preserve the existing DATA.update(... dropped_features=drop) interface.
    drop = []
    for col in test.select_dtypes(include=[np.number]):
        if col != "sample_id" and train[col].nunique() > 100:
            quantiles = np.linspace(0, 1, 41)
            bins = train[col].dropna().quantile(quantiles).dropna().unique()
            bins = np.sort(bins)
            train[col] = pd.cut(train[col], bins, include_lowest=True, labels=False)
            test[col] = pd.cut(test[col], bins, include_lowest=True, labels=False)
            test.loc[test[col].isna(), col] = 20

    train = reduce_mem_usage(train).fillna(0)
    test = reduce_mem_usage(test).fillna(0)

    CATS = [
        c for c in train.columns
        if train[c].dtype == "object"
        or train[c].dtype.name == "category"
        or train[c].nunique() <= 10
    ]
    NUMS = [c for c in train.columns if c not in CATS + ["sample_id", target_col]]

    print(f"categorical features: {len(CATS)}")
    print(f"numerical features  : {len(NUMS)}")

    for c in CATS:
        mapping = {v: i for i, v in enumerate(train[c].unique())}
        train[c] = train[c].map(mapping)
        test[c] = test[c].map(mapping).fillna(0)
    cat_dims = [train[c].nunique() for c in CATS]

    rssc = RobustScaleSmoothClipTransform()
    rssc.fit(train[NUMS].values)
    train[NUMS] = rssc.fit_transform(train[NUMS].values)
    test[NUMS] = rssc.transform(test[NUMS].values)

    # Exact original split: first 800k train, all remaining validation.
    train_size = 904390
    if len(train) <= train_size:
        raise ValueError(
            f"Need more than {train_size:,} rows; found {len(train):,}."
        )

    train_slice = train.iloc[:train_size]
    val_slice = train.iloc[train_size:]

    X_num_train = torch.tensor(train_slice[NUMS].values, dtype=torch.float32).to(device)
    X_cat_train = torch.tensor(train_slice[CATS].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(
        train_slice[target_col].round(4).values, dtype=torch.float32
    ).to(device)

    X_num_val = torch.tensor(val_slice[NUMS].values, dtype=torch.float32).to(device)
    X_cat_val = torch.tensor(val_slice[CATS].values, dtype=torch.float32).to(device)

    # Exact original behavior: validation target is NOT rounded.
    y_val = torch.tensor(
        val_slice[target_col].values, dtype=torch.float32
    ).to(device)

    print(f"training rows   : {len(train_slice):,}")
    print(f"validation rows : {len(val_slice):,}")
    print(f"training shape  : {tuple(X_num_train.shape)}")
    print(f"validation shape: {tuple(X_num_val.shape)}")

    DATA.update(dict(
        X_num_train=X_num_train, X_cat_train=X_cat_train, y_train=y_train,
        X_num_val=X_num_val, X_cat_val=X_cat_val, y_val=y_val,
        cat_dims=cat_dims, n_numerical=len(NUMS),
        CATS=CATS, NUMS=NUMS, dropped_features=drop,
    ))


# ===================================================================
# Optuna objective
# ===================================================================
def build_param_groups(model, lr, lr_scale_mult, lr_pbld_mult, lr_first_mult,
                        lr_bias_mult, weight_decay):
    scale_p, pbld_p, first_linear_p, other_w_p, bias_p = get_parameter_groups(model)
    return torch.optim.AdamW([
        {"params": scale_p, "lr": lr * lr_scale_mult, "weight_decay": weight_decay * 0.1},
        {"params": pbld_p, "lr": lr * lr_pbld_mult, "weight_decay": weight_decay},
        {"params": first_linear_p, "lr": lr * lr_first_mult, "weight_decay": weight_decay * 0.1},
        {"params": other_w_p, "lr": lr, "weight_decay": weight_decay},
        {"params": bias_p, "lr": lr * lr_bias_mult, "weight_decay": weight_decay * 0.5},
    ], betas=(0.9, 0.98))


def objective(trial: optuna.Trial, tune_epochs: int):
    set_seed(42)

    # ---- search space: tune only the top 10 parameters ----
    # Architecture is intentionally FIXED to match realmlp_capital.py exactly:
    #   n_ens=16, embed_dim=6, PBLD=(24, 3, 1.0), MLP=512->512->128, dropout=0.01,
    #   model RQ heads: n_rq_layers=2, rq_vocab_size=3.
    # Note: the original script builds 3 RQ target-code layers but the model has 2 RQ heads;
    # the loss therefore uses the first 2 code layers, exactly as in realmlp_capital.py.
    n_ens = 16
    embed_dim = 6
    model_n_rq_layers = 2
    rq_target_layers = 3
    rq_vocab_size = 3
    pbld_hidden_dim = 24
    pbld_out_dim = 3
    pbld_freq_scale = 1.0
    hidden_dims = (512, 512, 128)
    dropout = 0.01

    # Top 10 tuning parameters. These affect optimization/training only, not model structure.
    # Batch size is tuned around the original realmlp_capital.py value of 256.
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)
    lambda_rq = trial.suggest_float("lambda_rq", 1e-2, 5e-1, log=True)
    lambda_cos = trial.suggest_float("lambda_cos", 1e-3, 1e-1, log=True)
    ema_decay = trial.suggest_float("ema_decay", 0.99, 0.9995)
    lr_scale_mult = trial.suggest_float("lr_scale_mult", 10.0, 30.0)
    lr_pbld_mult = trial.suggest_float("lr_pbld_mult", 0.03, 0.30, log=True)
    lr_first_mult = trial.suggest_float("lr_first_mult", 0.5, 2.0, log=True)
    lr_bias_mult = trial.suggest_float("lr_bias_mult", 0.03, 0.30, log=True)
    train_bs = trial.suggest_categorical("train_bs", [128, 256, 512])

    # Keep gradient clipping identical to realmlp_capital.py.
    grad_clip = 1.0

    # ---- RQ target encoding: reproduce realmlp_capital.py ----
    y_train_np = DATA["y_train"].cpu().numpy()
    rq_encoder = RQKMeansEncoder(n_layers=rq_target_layers, codebook_size=rq_vocab_size)
    rq_encoder.fit(y_train_np.reshape(-1, 1))
    y_train_rq = torch.tensor(
        rq_encoder.encode(y_train_np.reshape(-1, 1)), dtype=torch.long
    ).to(device)

    model = RealMLP_RQ(
        output_dim=1, cat_dims=DATA["cat_dims"], n_numerical=DATA["n_numerical"],
        n_ens=n_ens, embed_dim=embed_dim,
        n_rq_layers=model_n_rq_layers, rq_vocab_size=rq_vocab_size,
        pbld_hidden_dim=pbld_hidden_dim, pbld_out_dim=pbld_out_dim,
        pbld_freq_scale=pbld_freq_scale, hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)

    optimizer = build_param_groups(
        model, lr,
        lr_scale_mult=lr_scale_mult,
        lr_pbld_mult=lr_pbld_mult,
        lr_first_mult=lr_first_mult,
        lr_bias_mult=lr_bias_mult,
        weight_decay=weight_decay,
    )

    # Original model always uses EMA; tune only its decay.
    ema = EMA(model, decay=ema_decay)

    # Keep LR multipliers in the same optimizer-group order used by build_param_groups().
    lr_multipliers = [lr_scale_mult, lr_pbld_mult, lr_first_mult, 1.0, lr_bias_mult]

    X_num_train, X_cat_train, y_train = DATA["X_num_train"], DATA["X_cat_train"], DATA["y_train"]
    X_num_val, X_cat_val, y_val = DATA["X_num_val"], DATA["X_cat_val"], DATA["y_val"]

    total_steps = (len(y_train) + train_bs - 1) // train_bs * tune_epochs
    best_val_cos = -1.0

    for epoch in range(tune_epochs):
        model.train()
        perm = torch.randperm(len(y_train))
        X_num_s, X_cat_s = X_num_train[perm], X_cat_train[perm]
        y_s, y_rq_s = y_train[perm], y_train_rq[perm]

        for i in range(0, len(y_train), train_bs):
            batch_idx = i // train_bs
            global_step = epoch * ((len(y_train) + train_bs - 1) // train_bs) + batch_idx
            progress = min(global_step / total_steps, 1.0)

            for pg, mult in zip(optimizer.param_groups, lr_multipliers):
                pg["lr"] = flat_anneal(lr * mult, progress)

            batch_x_num = X_num_s[i:i + train_bs]
            batch_x_cat = X_cat_s[i:i + train_bs]
            batch_y = y_s[i:i + train_bs]
            batch_y_rq = y_rq_s[i:i + train_bs]
            noise_std = 0.005 * (1 - progress)
            batch_y_noisy = batch_y + torch.randn_like(batch_y) * noise_std

            optimizer.zero_grad()
            code_logits, y_pred = model(batch_x_num, batch_x_cat, return_codes=True)
            loss, cos_sim, mse_loss, rq_loss = compute_loss_with_rq(
                y_pred, batch_y_noisy, code_logits, batch_y_rq,
                lambda_cos=lambda_cos, lambda_rq=flat_anneal(lambda_rq, progress),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update()

        original_state = ema.apply() if ema is not None else None
        val_cos, _ = evaluate_model(model, X_num_val, X_cat_val, y_val)
        if ema is not None:
            ema.restore(original_state)

        best_val_cos = max(best_val_cos, val_cos)
        trial.report(val_cos, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_cos


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for RealMLP_RQ")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=13*3600, help="seconds, overall study timeout")
    parser.add_argument("--tune-epochs", type=int, default=10, help="epochs per trial; 10 matches the original model")
    parser.add_argument("--study-name", type=str, default="realmlp_exact_baseline_20260924")
    parser.add_argument("--storage", type=str, default="sqlite:///realmlp_exact_baseline.db")
    parser.add_argument("--n-startup-trials", type=int, default=10)
    args = parser.parse_args()

    #load_data(args.tune_train_size, args.tune_val_size)
    load_data()

    sampler = TPESampler(seed=42, n_startup_trials=args.n_startup_trials)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    print("\nSkipping original baseline run; starting Optuna tuning immediately.")

    study.optimize(
        lambda trial: objective(trial, args.tune_epochs),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    print("\n" + "=" * 60)
    print(f"Best val cosine similarity: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = "realmlp_capital_best_params_20260924.csv"
    with open(out_path, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\nSaved best params to {out_path}")


if __name__ == "__main__":
    main()
