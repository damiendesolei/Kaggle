# -*- coding: utf-8 -*-
"""
Created on Tue May  6 22:27:07 2025

@author: zrj-desktop
"""

import pandas as pd
import numpy as np


PATH = r'G:\\kaggle\user-retention-prediction\submisssion\voting\\'

sub_1 = pd.read_csv(PATH+'TimeSeries_is_about_the_lines_36.7.csv') # On Kaggle
sub_2 = pd.read_csv(PATH+'Cbt_MultiClass_smape_39.4.csv') # Local
sub_3 = pd.read_csv(PATH+'model_xgb_model_1_sub_43.1.csv') # On Kaggle


sub_1.rename(columns={"pred": "pred_1"}, inplace=True)
sub_2.rename(columns={"pred": "pred_2"}, inplace=True)
sub_3.rename(columns={"pred": "pred_3"}, inplace=True)

sub = pd.merge(sub_1, sub_2, on='ID', how='inner')
sub = pd.merge(sub, sub_3, on='ID', how='inner')


def weighted_avg_rounding(df, columns, weights):
    """
    Performs weighted average + rounding ensemble on specified columns of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with model predictions.
        columns (list): List of column names containing model predictions.
        weights (list or np.array): Same length as columns. Will be normalized if they don't sum to 1.

    Returns:
        pd.Series: Final ensemble predictions as integers.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights

    preds = df[columns].values  # Shape: [n_samples, n_models]
    weighted_avg = np.dot(preds, weights)
    rounded_preds = np.rint(weighted_avg).astype(int)

    # Optional: clip to ordinal range (0 to 7)
    return pd.Series(np.clip(rounded_preds, 0, 7), index=df.index)



# Weight 1 - LB 35.54 
model_weights = np.array([45, 35, 20])

pred_voted = weighted_avg_rounding(sub, ["pred_1", "pred_2", "pred_3"], model_weights)
sub['pred'] = pred_voted

thresholds = np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806]) 
print(sub.pred.value_counts(normalize=True).sort_index(ascending=False))
#sub.to_csv(PATH+'submission_votee_all.csv', index=False)
sub[['ID','pred']].to_csv(PATH+'submission_votee_45_35_20.csv', index=False)


# Weight 2 - LB 36.27
model_weights = np.array([33, 33, 33])

pred_voted = weighted_avg_rounding(sub, ["pred_1", "pred_2", "pred_3"], model_weights)
sub['pred'] = pred_voted

thresholds = np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806]) 
print(sub.pred.value_counts(normalize=True).sort_index(ascending=False))
#sub.to_csv(PATH+'submission_votee_all.csv', index=False)
sub[['ID','pred']].to_csv(PATH+'submission_voted_33_33_33.csv', index=False)


# Weight 3 - LB 35.66
model_weights = np.array([60, 40, 0])

pred_voted = weighted_avg_rounding(sub, ["pred_1", "pred_2", "pred_3"], model_weights)
sub['pred'] = pred_voted

thresholds = np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806]) 
print(sub.pred.value_counts(normalize=True).sort_index(ascending=False))
#sub.to_csv(PATH+'submission_votee_all.csv', index=False)
sub[['ID','pred']].to_csv(PATH+'submission_voted_60_40_0.csv', index=False)