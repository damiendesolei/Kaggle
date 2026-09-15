# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:31:09 2026

@author: azz
"""

import pandas as pd
import numpy as np
#from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


DATA_DIR =  r"H:\kaggle\ms-capital-real-financial-market-forecasting"

files = {
    "label": DATA_DIR + "\\train\\label.feather",
    "market": DATA_DIR +"\\train\\market.feather",
    "order": DATA_DIR + "\\train\\order.feather",
    "transaction": DATA_DIR  + "\\train\\transaction.feather",
}



def profile_df(df, name):
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing %:")
    print(
        (df.isna().mean() * 100)
        .sort_values(ascending=False)
        .round(2)
    )

    print("\nUnique values:")
    print(df.nunique().sort_values())

    print("\nMemory:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")
    
    
    
################ label file ###############    
label = pd.read_feather(files["label"])
# profile_df(label, "LABEL")

# label.describe(include="all")
# label.groupby("month")["target"].agg(
#     ["count", "mean", "std", "min", "median", "max"]
# )

# label.groupby("month").agg(
#     n_samples=("sample_id", "size"),
#     min_sample_id=("sample_id", "min"),
#     max_sample_id=("sample_id", "max"),
# )

# label["sample_id"].is_unique
# label["sample_id"].duplicated().sum()
# label["month"].value_counts().sort_index()




################ Market ###############
market = pd.read_feather(files["market"])
# profile_df(market, "MARKET")

# market.groupby("sample_id").size().describe()
# market.groupby("sample_id")["seconds_before_predict"].agg(
#     ["count", "min", "max"]
# ).describe()

# market = market.sort_values(
#     ["sample_id", "seconds_before_predict"],
#     ascending=[True, False]
# )

# market["dt"] = (
#     market.groupby("sample_id")["seconds_before_predict"]
#     .diff()
#     .abs()
# )

# market["dt"].describe()


# Build once, outside the loop/function
market_label = market.merge(
    label[["sample_id", "target"]],
    on="sample_id",
    how="left"
)
market_label["_mid"] = (market_label["ask_price_1"] + market_label["bid_price_1"]) * 0.5


def plot_sample(sample_id, max_seconds_before_predict=None, market_label=market_label):
    """Plot _mid vs seconds_before_predict for a single sample_id.

    max_seconds_before_predict: if given, only keep rows where
        seconds_before_predict <= this value.
    """
    plot_df = (
        market_label[market_label["sample_id"] == sample_id]
        .sort_values("seconds_before_predict", ascending=False)
    )

    if max_seconds_before_predict is not None:
        plot_df = plot_df[plot_df["seconds_before_predict"] <= max_seconds_before_predict]

    if plot_df.empty:
        raise ValueError(
            f"No rows found for sample_id={sample_id}"
            + (f" with seconds_before_predict <= {max_seconds_before_predict}"
               if max_seconds_before_predict is not None else "")
        )

    target = plot_df["target"].iloc[0]
    line_color = "blue" if target >= 0 else "red"

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.invert_xaxis()

    ax1.plot(
        plot_df["seconds_before_predict"],
        plot_df["_mid"],
        marker="o",
        markersize=3,
        color=line_color,
        label="_mid"
    )

    ax1.set_xlabel("seconds_before_predict")
    ax1.set_ylabel("_mid")
    ax1.set_title(f"Sample ID = {sample_id}, Target = {target:.6f}")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.show()
    return plot_df



# Plot out each sample_id
for i in range(500, 1000):
    try:
        plot_sample(i, 100)
    except Exception as e:
        print(f"Skipping sample_id={i}: {e}")
        continue