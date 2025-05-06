# -*- coding: utf-8 -*-
"""
Created on Mon May  5 22:53:49 2025

@author: zrj-desktop
"""

import pandas as pd
import numpy as np


PATH = r'G:\\kaggle\user-retention-prediction\submisssion\\'

sub_1 = pd.read_csv(PATH+'TimeSeries_is_about_the_lines_36.7_raw.csv', usecols=['ID','pred_raw']) # On Kaggle
sub_2 = pd.read_csv(PATH+'Cbt_MultiClass_smape_39.4_raw.csv') # Local
sub_3 = pd.read_csv(PATH+'model_xgb_model_1_sub_43.1_raw.csv') # On Kaggle

sub_1.rename(columns={"pred_raw": "pred_1"}, inplace=True)
sub_2.rename(columns={"pred": "pred_2"}, inplace=True)
sub_3.rename(columns={"pred": "pred_3"}, inplace=True)

sub = pd.merge(sub_1, sub_2, on='ID', how='inner')
sub = pd.merge(sub, sub_3, on='ID', how='inner')


thresholds = np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806]) 
#sub = pd.DataFrame()


# 43.02
sub['avg_pred'] = 0.6*sub_1['pred_1'] + 0.3*sub_2['pred_2'] + 0.1*sub_3['pred_3']
sub["avg_rank"]=sub["avg_pred"].rank()
sub["avg_rank_normalized"]=sub["avg_rank"]/(sub["avg_rank"].max())
sub["pred"]=np.digitize(sub["avg_rank_normalized"], thresholds).clip(0,7)

print(sub.pred.value_counts(normalize=True).sort_index(ascending=False))
sub[['ID','pred']].to_csv(PATH+'submission_ensemble_6_3_1.csv', index=False)


# 46.06
sub['avg_pred'] = sub_1['pred_1']/3 + sub_2['pred_2']/3 + sub_3['pred_3']/3
sub["avg_rank"]=sub["avg_pred"].rank()
sub["avg_rank_normalized"]=sub["avg_rank"]/(sub["avg_rank"].max())
sub["pred"]=np.digitize(sub["avg_rank_normalized"], thresholds).clip(0,7)

sub.pred.value_counts(normalize=True)
sub[['ID','pred']].to_csv(PATH+'submission_ensemble_equal_weight.csv', index=False)


