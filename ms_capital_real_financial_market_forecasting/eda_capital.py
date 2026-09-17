# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:31:09 2026

@author: azz
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


DATA_DIR = r"H:\kaggle\ms-capital-real-financial-market-forecasting"

files = {
    "label": DATA_DIR + "\\train\\label.feather",
    "market": DATA_DIR + "\\train\\market.feather",
    "order": DATA_DIR + "\\train\\order.feather",
    "transaction": DATA_DIR + "\\train\\transaction.feather",
}

label = pd.read_feather(files["label"])
market = pd.read_feather(files["market"])
extract_id = 802
#market[market.sample_id==extract_id].to_csv(f"H:\kaggle\ms-capital-real-financial-market-forecasting\processed_data\market_sample_id_{extract_id}.csv", index=False)

market_label = market.merge(
    label[["sample_id", "target"]], on="sample_id", how="left"
)
market_label["_mid"] = (
    market_label["ask_price_1"] + market_label["bid_price_1"]
) * 0.5


def fit_linear_regression(x, y):
  """Closed-form OLS linear regression: y = intercept + slope * x."""
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  x_mean, y_mean = x.mean(), y.mean()
  denom = np.sum((x - x_mean) ** 2)
  slope = np.sum((x - x_mean) * (y - y_mean)) / denom if denom > 0 else 0.0
  intercept = y_mean - slope * x_mean
  return slope, intercept


def plot_sample(
    sample_id,
    max_seconds_before_predict=None,
    fit_max_seconds=60,  # changed threshold parameter to capture <= 60s
    market_label=market_label,
):
  plot_df = (
      market_label[market_label["sample_id"] == sample_id]
      .sort_values("seconds_before_predict", ascending=False)
      .copy()
  )

  if max_seconds_before_predict is not None:
    plot_df = plot_df[
        plot_df["seconds_before_predict"] <= max_seconds_before_predict
    ]

  if plot_df.empty:
    raise ValueError(f"No rows found for sample_id={sample_id}")

  target = plot_df["target"].iloc[0]
  line_color = "blue" if target >= 0 else "red"

  # --- linear regression fit on seconds_before_predict <= fit_max_seconds ---
  fit_df = plot_df[
      plot_df["seconds_before_predict"] <= fit_max_seconds
  ].copy()

  reg_line = None
  slope = None
  predicted_at_0 = None
  actual_time = None
  actual_mid = None
  reg_line_color = "red"

  if len(fit_df) >= 2:
    # Fit OLS using seconds_before_predict <= 60 and _mid
    slope, intercept = fit_linear_regression(
        fit_df["seconds_before_predict"], fit_df["_mid"]
    )
    
    # Set regression line color based on slope sign (red if negative, blue otherwise)
    reg_line_color = "blue" if slope < 0 else "red"
    
    # Prediction at seconds_before_predict = 0
    predicted_at_0 = intercept + slope * 0

    # Generate regression line from the max of the fit window down to 0
    reg_sec = np.linspace(fit_df["seconds_before_predict"].max(), 0, 100)
    reg_y = intercept + slope * reg_sec
    reg_line = pd.DataFrame({"seconds_before_predict": reg_sec, "fitted": reg_y})

    # closest actual observation to the prediction point, for comparison
    actual_row = plot_df.loc[plot_df["seconds_before_predict"].idxmin()]
    actual_time = actual_row["seconds_before_predict"]
    actual_mid = actual_row["_mid"]

  fig, ax1 = plt.subplots(figsize=(12, 6))
  
  # Reverse/invert the x-axis so countdown time runs right-to-left
  ax1.invert_xaxis()

  ax1.plot(
      plot_df["seconds_before_predict"],
      plot_df["_mid"],
      marker="o",
      markersize=3,
      color=line_color,
      label="_mid (actual)",
  )

  if reg_line is not None:
    ax1.plot(
        reg_line["seconds_before_predict"],
        reg_line["fitted"],
        linestyle="--",
        color=reg_line_color,  # Dynamic color based on slope
        linewidth=1.5,
        label=f"linear fit (seconds <= {fit_max_seconds}, slope={slope:.6f})",
    )

    ax1.scatter([0], [predicted_at_0], marker="x", s=100, color="orange", zorder=5)

    # Display slope and predictions clearly in the annotation box
    annotation = f"slope = {slope:.6f}\npredicted @0 = {predicted_at_0:.6f}"
    if actual_time is not None:
      annotation += f"\nactual @{actual_time:.1f}s = {actual_mid:.6f}"
      if actual_time <= 5.0:
        annotation += f"\nresid = {actual_mid - predicted_at_0:+.6f}"
      ax1.scatter(
          [actual_time], [actual_mid], marker="x", s=100, color="green", zorder=5
      )

    ax1.annotate(
        annotation,
        xy=(0, predicted_at_0),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
    )

  ax1.set_xlabel("seconds_before_predict")
  ax1.set_ylabel("_mid")
  ax1.set_title(f"Sample ID = {sample_id}, Target = {target:.6f}")
  ax1.grid(True, alpha=0.3)
  ax1.xaxis.set_major_locator(mticker.MultipleLocator(5))
  ax1.legend(loc="best", fontsize=8)

  plt.show()
  return plot_df


# Plot out each sample_id
for i in range(800, 850):
  try:
    plot_sample(i, 100)
  except Exception as e:
    print(f"Skipping sample_id={i}: {e}")
    continue