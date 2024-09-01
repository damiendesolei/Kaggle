# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:52:43 2024

@author: zrj-desktop
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
#import pandas.api.types
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold

import lightgbm as lgb
import xgboost
import optuna
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_optimization_history

import re

from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import torch 
import cv2
import os, io
from io import BytesIO

from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.cuda import amp
import torchvision
from torcheval.metrics.functional import binary_auroc

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import sys
import time
import math
import copy
import gc
from tqdm import tqdm

from PIL import Image

import albumentations as A

import math, random
import glob
import h5py

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"



##############################################
######### read in data #######################
PATH = r'G:\\kaggle\isic-2024-challenge\\'

train = pd.read_csv(PATH + 'train-metadata.csv', low_memory=False)
test = pd.read_csv(PATH + 'test-metadata.csv', low_memory=False)
submission = pd.read_csv(PATH + 'sample_submission.csv', low_memory=False)
######### finish reading in data #############
##############################################



##############################################
######### glimpse of data ####################
train.info()
train.describe().T
train.target.value_counts(normalize=True)
train.sex.value_counts(normalize=True)
train.tbp_tile_type.value_counts(normalize=True)
train.tbp_lv_location.value_counts(normalize=True)
train.tbp_lv_location_simple.value_counts(normalize=True)
train.anatom_site_general.value_counts(normalize=True)



##############################################
######### feature enginerring ################
#https://www.kaggle.com/code/snnclsr/lgbm-baseline-with-new-features
def feature_engineering(df):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + \
                                        df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    # https://www.kaggle.com/code/dschettler8845/isic-detect-skin-cancer-let-s-learn-together
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(df["tbp_lv_x"]**2 + df["tbp_lv_y"]**2 + df["tbp_lv_z"]**2)
    df["color_range"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]

    new_num_cols = [
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",
        
        "color_variance_ratio", "border_color_interaction", "size_color_contrast_ratio",
        "age_normalized_nevi_confidence", "color_asymmetry_index", "3d_volume_approximation",
        "color_range", "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index",
    ]
    new_cat_cols = ["combined_anatomical_site"]
    return df, new_num_cols, new_cat_cols


train, new_num_cols, new_cat_cols = feature_engineering(train.copy())
test, _, _ = feature_engineering(test.copy())
    
num_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
    ] + new_num_cols

# anatom_site_general
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"] + new_cat_cols
train_cols = num_cols + cat_cols

# category_encoder = OrdinalEncoder(
#     categories='auto',
#     dtype=int,
#     handle_unknown='use_encoded_value',
#     unknown_value=-2,
#     encoded_missing_value=-1,
# )

# X_cat = category_encoder.fit_transform(df_train[cat_cols])
# for c, cat_col in enumerate(cat_cols):
#     df_train[cat_col] = X_cat[:, c]





##############################################
######### encoding object columns ############
#integer encode the sex
def int_encode_sex(col):
    if col == 'male':
        return 0
    elif col == 'female':
        return 1
    else:
        return -1
#train.groupby('Gender').count()
train['sex_IntEncoded'] = train['sex'].apply(int_encode_sex)
test['sex_IntEncoded'] = test['sex'].apply(int_encode_sex)
test[['sex','sex_IntEncoded']].drop_duplicates() #show encoding


#integer encode the tbp_tile_type
def int_encode_tbp_tile_type(col):
    if col == '3D: XP':
        return 0
    elif col == '3D: white':
        return 1
    else:
        return -1
#train.groupby('Gender').count()
train['tbp_tile_type_IntEncoded'] = train['tbp_tile_type'].apply(int_encode_tbp_tile_type)
test['tbp_tile_type_IntEncoded'] = test['tbp_tile_type'].apply(int_encode_tbp_tile_type)
test[['tbp_tile_type','tbp_tile_type_IntEncoded']].drop_duplicates() #show encoding


#extract location from tbp_lv_location
def extract_location(row):
    match1 = re.search(r'(Top|top)', str(row['tbp_lv_location']).lower())   
    match2 = re.search(r'(Middle|middle)', str(row['tbp_lv_location']).lower())
    match3 = re.search(r'(Bottom|bottom)', str(row['tbp_lv_location']).lower())
    match4 = re.search(r'(Upper|upper)', str(row['tbp_lv_location']).lower())
    match5 = re.search(r'(Lower|lower)', str(row['tbp_lv_location']).lower())
    if match1:
        return 'top'
    elif match2:
        return 'middle'
    elif match3:
        return 'bottom'
    elif match4:
        return 'upper'
    elif match5:
        return 'lower'    
    else:
        return 'other'    
train['location'] = train.apply(extract_location, axis=1)
test['location'] = test.apply(extract_location, axis=1)   
train[['tbp_lv_location','location']].drop_duplicates() #show mapping
#integer encode location
def int_encode_location(col):
    if col == 'top':
        return 0
    elif col == 'upper':
        return 1
    elif col == 'lower':
        return 2
    elif col == 'middle':
        return 3
    elif col == 'bottom':
        return 4
    elif col == 'other':
        return 5
    else:
        return -1
#train.groupby('Gender').count()
train['location_IntEncoded'] = train['location'].apply(int_encode_location)
test['location_IntEncoded'] = test['location'].apply(int_encode_location)
train[['location','location_IntEncoded']].drop_duplicates() #show encoding


#integer encode tbp_lv_location_simple
def int_encode_tbp_lv_location_simple(col):
    if col == 'Head & Neck':
        return 0
    elif col == 'Torso Back':
        return 1
    elif col == 'Torso Front':
        return 2
    elif col == 'Left Leg':
        return 3
    elif col == 'Right Leg':
        return 4
    elif col == 'Left Arm':
        return 5
    elif col == 'Right Arm':
        return 6
    elif col == 'Unknown':
        return 7
    else:
        return -1
#train.groupby('Gender').count()
train['tbp_lv_location_simple_IntEncoded'] = train['tbp_lv_location_simple'].apply(int_encode_tbp_lv_location_simple)
test['tbp_lv_location_simple_IntEncoded'] = test['tbp_lv_location_simple'].apply(int_encode_tbp_lv_location_simple)
train[['tbp_lv_location_simple','tbp_lv_location_simple_IntEncoded']].drop_duplicates() #show encoding


# extract position from combined_anatomical_site
def extract_position(row):
    match1 = re.search(r'(Extremity|extremity)', str(row['combined_anatomical_site']).lower())   
    match2 = re.search(r'(Posterior|posterior)', str(row['combined_anatomical_site']).lower())
    match3 = re.search(r'(Anterior|anterior)', str(row['combined_anatomical_site']).lower())
    if match1:
        return 'extremity'
    elif match2:
        return 'posterior'
    elif match3:
        return 'anterior' 
    else:
        return 'other'    
train['position'] = train.apply(extract_position, axis=1)
test['position'] = test.apply(extract_position, axis=1)   
train[['combined_anatomical_site','position']].drop_duplicates() #show mapping
#integer encode position
def int_encode_position(col):
    if col == 'extremity':
        return 0
    elif col == 'posterior':
        return 1
    elif col == 'anterior':
        return 2
    elif col == 'other':
        return 3
    else:
        return -1
#train.groupby('Gender').count()
train['position_IntEncoded'] = train['position'].apply(int_encode_position)
test['position_IntEncoded'] = test['position'].apply(int_encode_position)
train[['position','position_IntEncoded']].drop_duplicates() #show encoding


##############################################
######### feature probing - EDA ##############
# sns.kdeplot(data=train, x='sex_IntEncoded', hue='target')
# sns.countplot(data=train, x='target', hue='sex')
# sns.countplot(data=train[train.target==1], x='target', hue='sex') #For target=1, 'male' has a higher number

# sns.kdeplot(data=train, x='tbp_tile_type_IntEncoded', hue='target')
# sns.countplot(data=train, x='target', hue='tbp_tile_type')
# sns.countplot(data=train[train.target==0], x='target', hue='tbp_tile_type') #For target==0, '3D: XP' has a higher number

# #sns.kdeplot(data=train, x='tbp_lv_location_Intencoded', hue='target');
# sns.countplot(data=train, x='target', hue='tbp_lv_location')
# sns.countplot(data=train[train.target==1], x='target', hue='tbp_lv_location')

# sns.kdeplot(data=train, x='location_Intencoded', hue='target');
# sns.countplot(data=train, x='target', hue='location')

# sns.kdeplot(data=train, x='position_IntEncoded', hue='target');
# sns.countplot(data=train, x='target', hue='position')

######### End of feature probing EDA #########
##############################################




##############################################
######### modelling setup  ###################

def custom_loss_func(y_actual_, y_predicted_):
    v_gt = abs(np.asarray(y_actual_)-1)
    v_pred = np.array([1.0 - x for x in y_predicted_])
    max_fpr = abs(1-0.80)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

custom_score = make_scorer(custom_loss_func, greater_is_better=True)


# https://www.kaggle.com/code/snnclsr/lgbm-baseline-with-new-features
def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)


initial_features = num_cols + ['sex_IntEncoded', 'tbp_tile_type_IntEncoded', 'location_IntEncoded', 'tbp_lv_location_simple_IntEncoded', 'position_IntEncoded']

def cross_validate(model, label, features=initial_features):
    """Compute out-of-fold and test predictions for a given model.
    
    Out-of-fold and test predictions are stored in the global variables
    oof and test_pred, respectively. 
    """
    start_time = datetime.datetime.now()
    tr_scores = []
    va_scores = []
    oof_preds = np.full_like(train.target, np.nan, dtype=np.float64)
    #for fold, (idx_tr, idx_va) in enumerate(kf.split(train)):
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.target)):    
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train.target[idx_tr]
        y_va = train.target[idx_va]
        
        model.fit(X_tr, y_tr)
        #y_pred = model.predict(X_va)
        y_pred = model.predict_proba(X_va)
        #y_predicted = np.argmax(y_pred, axis=1) #find the highest probability row array position 
        y_predicted = y_pred[:,1] #find the probability of being 1
        

        #va_score = partial_auc(y_va, y_predicted)
        va_score = comp_score(y_va, pd.DataFrame(y_predicted), "")
        #tr_score = roc_auc_score(y_tr, model.predict(X_tr))
        #tr_score = roc_auc_score(y_tr, np.argmax(model.predict_proba(X_tr), axis=1))
        #tr_score = roc_auc_score(y_tr, model.predict_proba(X_tr)[:,1])
        
        tr_score = comp_score(y_tr, pd.DataFrame(model.predict_proba(X_tr)[:,1]), "")
        print(f"# Fold {fold}: tr_auc={tr_score:.5f}, val_auc={va_score:.5f}")
        va_scores.append(va_score)
        tr_scores.append(tr_score)
        oof_preds[idx_va] = y_predicted #each iteration will fill in 1/5 of the index
        #oof_preds[idx_va] = y_pred
            
    elapsed_time = datetime.datetime.now() - start_time
    print(f"{Fore.RED}# Overall val={np.array(va_scores).mean():.5f} {label}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    print(f"{Fore.RED}# {label} Fitting started from {start_time}")
    oof[label] = oof_preds

    if COMPUTE_HOLDOUT_PRED:
        X_ho = holdout[features]
        y_ho = holdout.target
        model.fit(X_ho, y_ho)
        y_pred = model.predict_proba(holdout[features])
        #y_predicted = np.argmax(y_pred, axis=1)
        y_predicted = y_pred[:,1]
        #holdout_pred[label] = y_predicted
        #ho_score = roc_auc_score(holdout.Response, y_predicted)
        ho_score = comp_score(holdout.target, pd.DataFrame(y_predicted), "")
        print('# Holdout score is: ' + str(ho_score))
 
    if COMPUTE_TEST_PRED:
        X_tr = train[features]
        y_tr = train.target
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(test[features])
        #y_predicted = np.argmax(y_pred, axis=1)
        y_predicted = y_pred[:,1]
        test_pred[label] = y_predicted
        return test_pred
    
# want to see the cross-validation results)
COMPUTE_TEST_PRED = True
COMPUTE_HOLDOUT_PRED = False

# Containers for results
oof, test_pred, holdout_pred = {}, {}, {}
######### finish modelling setup ##############
###############################################



##############################################
######### baseline model #####################



#lgb ~ 1m
lgb_model = lgb.LGBMClassifier(verbose=-1, eval_metric='custom_score', device='cpu')
cross_validate(lgb_model, 'LightGBM_Untuned', features=initial_features)
# Fold 0: tr_auc=0.01901, val_auc=0.01650
# Fold 1: tr_auc=0.02095, val_auc=0.02046
# Fold 2: tr_auc=0.03968, val_auc=0.02996
# Fold 3: tr_auc=0.02669, val_auc=0.02944
# Fold 4: tr_auc=0.02709, val_auc=0.02500
# Overall val=0.02427 LightGBM_Untuned   0 min
# LightGBM_Untuned Fitting started from 2024-08-22 20:07:46.728607


#lgb - tuned
def objective(trial):
    X_train, X_valid, y_train, y_valid = train_test_split(train[initial_features], train.target, test_size=0.3)

    param = {
             #"objective": "CrossEntropy",
             "iterations": trial.suggest_int("iterations", 800, 2000),
             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
             "depth": trial.suggest_int("depth", 5, 25),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1000, 20000),
             'subsample': trial.suggest_float('subsample', 0, 1),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
             'max_bin': trial.suggest_int("max_bin", 8000, 200000),
             'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 20, log=True),
             'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 20, log=True),
             'pos_bagging_fraction': trial.suggest_float('pos_bagging_fraction', 0, 1),
             'neg_bagging_fraction': trial.suggest_float('neg_bagging_fraction', 0, 1),
             #used_ram_limit": "48gb",
             "eval_metric": 'custom_score',
             "device": 'cpu'
            }


    gbm = lgb.LGBMClassifier(**param)

    gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    #y_preds = gbm.predict(X_valid)
    y_preds = gbm.predict_proba(X_valid)[:,1]
    #pred_labels = np.rint(preds)
    #score = comp_score(y_valid, y_preds)
    score = comp_score(y_valid, pd.DataFrame(y_preds), "")
    return score
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=200, timeout=3600)
#study_summaries = optuna.study.get_all_study_summaries()
# Trial 0 finished with value: 0.17058399728141566 and parameters: {'iterations': 1564, 'learning_rate': 0.021861424132643192, 'colsample_bylevel': 0.09839000386075163, 'depth': 15, 'min_data_in_leaf': 4169, 'subsample': 0.24265885764272066, 'colsample_bytree': 0.40469534371300453, 'max_bin': 15603, 'reg_lambda': 1.6371338817060606e-05, 'reg_alpha': 4.4221534988928415, 'pos_bagging_fraction': 0.5623107344062629, 'neg_bagging_fraction': 0.7514980166710389}. Best is trial 0 with value: 0.17058399728141566.
#Trial 21 finished with value: 0.1715800231544985 and parameters: {'iterations': 1401, 'learning_rate': 0.05032974645541588, 'colsample_bylevel': 0.02898675577640679, 'depth': 18, 'min_data_in_leaf': 17839, 'subsample': 0.8920051990627909, 'colsample_bytree': 0.6202745141075666, 'max_bin': 125990, 'reg_lambda': 0.002175710371681295, 'reg_alpha': 2.504363052953543e-05, 'pos_bagging_fraction': 0.14508814884263116, 'neg_bagging_fraction': 0.4948871177523545}. Best is trial 21 with value: 0.1715800231544985. 
# plotly_config = {"staticPlot": True}
# fig = plot_optimization_history(study)
# fig.show(config=plotly_config)
# fig = plot_param_importances(study)
# fig.show(config=plotly_config)


#lgb - tuned ~ 2m
params = {'iterations': 1401, 'learning_rate': 0.05032974645541588, 'colsample_bylevel': 0.02898675577640679, 'depth': 18, 'min_data_in_leaf': 17839, 'subsample': 0.8920051990627909, 'colsample_bytree': 0.6202745141075666, 'max_bin': 125990, 'reg_lambda': 0.002175710371681295, 'reg_alpha': 2.504363052953543e-05, 'pos_bagging_fraction': 0.14508814884263116, 'neg_bagging_fraction': 0.4948871177523545}
lgb_model = lgb.LGBMClassifier(**params, verbose=-1, eval_metric='custom_score', device='cpu')
cross_validate(lgb_model, 'LightGBM_Tuned', features=initial_features)
# Fold 0: tr_auc=0.18532, val_auc=0.16740
# Fold 1: tr_auc=0.18571, val_auc=0.15080
# Fold 2: tr_auc=0.18670, val_auc=0.15552
# Fold 3: tr_auc=0.18603, val_auc=0.16114
# Fold 4: tr_auc=0.18735, val_auc=0.13777
# Overall val=0.15453 LightGBM_Tuned   2 min
# LightGBM_Tuned Fitting started from 2024-08-22 20:12:48.521399
model = lgb_model
importances = model.feature_importances_ 
df_imp = pd.DataFrame({"feature": model.feature_name_, "importance": importances}).sort_values("importance").reset_index(drop=True)

plt.figure(figsize=(16, 12))
plt.barh(df_imp["feature"], df_imp["importance"])
plt.show()





##############################################
################## NN ########################

# Create the model without loading pretrained weights and save to a specific location
model = timm.create_model("hf_hub:timm/efficientnet_b3.ra2_in1k", pretrained=True)
SAVE_PATH = r'G:\\kaggle\isic-2024-challenge\efficientnet_b3\pytorch\efficientnet_b3_ra2_in1k.pth'
torch.save(model.state_dict(), SAVE_PATH)


# https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only
CONFIG = {
    "seed": 24,
    "epochs": 25,
    "img_size": 384,
    "model_name": "hf_hub:timm/efficientnet_b3.ra2_in1k",
    #"checkpoint_path" : "/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b0/1/tf_efficientnet_b0_aa-827b6e33.pth",
    "checkpoint_path" : PATH + "efficientnet_b3\\pytorch\\efficientnet_b3_ra2_in1k.pth",
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 1e-3,#1e-4,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-6,
    "fold" : 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Specify the model architecture
#model = timm.create_model("hf_hub:timm/resnet50d.ra4_e3600_r224_in1k", pretrained=True)


#seet seed for reproductivity
def set_seed(seed=24):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
      
set_seed(CONFIG['seed'])



#ROOT_DIR = "/kaggle/input/isic-2024-challenge"
TRAIN_DIR = PATH + 'train-image\\image'
def get_train_file_path(image_id):
    #return f"{TRAIN_DIR}/{image_id}.jpg"
    return f"{TRAIN_DIR}\\{image_id}.jpg"



#read in data
#train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images = sorted(glob.glob(f"{TRAIN_DIR}\\*.jpg"))

df = train

df.shape
df.target.value_counts(normalize=True)

#rebalance data
df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*19, :]])  # positive% is set to 5%
df.shape
df.target.value_counts(normalize=True)

df['file_path'] = df['isic_id'].apply(get_train_file_path)
df = df[df["file_path"].isin(train_images)].reset_index(drop=True)
df.head()

#set up t max
CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
CONFIG['T_max']

# create folds
sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])

for fold, ( _, val_) in enumerate(sgkf.split(df, df.target,df.patient_id)):
      df.loc[val_ , "kfold"] = int(fold)
      
      
#dataset classes
class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }    
    
  



#Augmentation
data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


# import timm
# CONFIG['model_name']
# model = timm.create_model(CONFIG['model_name'], pretrained=False)
# model.eval();
# config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**config, is_training=False)
# print(config)
# print(transforms)



#GeM pooling
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
                
                          


#Create the model
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

model = ISICModel(CONFIG['model_name'], checkpoint_path=CONFIG['checkpoint_path'])
model.to(CONFIG['device']);



#Loss function
def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)




#Train the model
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0) #avoid error "Assertion `input_val >= zero && input_val <= one` failed."
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc



#Validation model
@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0) #avoid error "Assertion `input_val >= zero && input_val <= one` failed."
        loss = criterion(outputs, targets)

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss, epoch_auroc



#Run training
def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("Cuda available - Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['lr'].append( scheduler.get_lr()[0] )
        
        # deep copy the model
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            #PATH_ = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            temp = r'G:\\kaggle\isic-2024-challenge\efficientnet_b3\pytorch'
            PATH = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            PATH_ = temp + '\\' + PATH
            print("Model is saved to: " + str(PATH_))
            torch.save(model.state_dict(), PATH_)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset_for_Train(df_train, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=0, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


train_loader, valid_loader = prepare_loaders(df, fold=CONFIG["fold"])

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)

model, history = run_training(model, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=CONFIG['epochs'])


#History
history = pd.DataFrame.from_dict(history)
history.to_csv("history.csv", index=False)

#Logs
plt.plot( range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
plt.plot( range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

#Loss plot 
plt.plot( range(history.shape[0]), history["Train AUROC"].values, label="Train AUROC")
plt.plot( range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
plt.xlabel("epochs")
plt.ylabel("AUROC")
plt.grid()
plt.legend()
plt.show()

#Learning rate
plt.plot( range(history.shape[0]), history["lr"].values, label="lr")
plt.xlabel("epochs")
plt.ylabel("lr")
plt.grid()
plt.legend()
plt.show()

################## End of NN #################
##############################################




##############################################
################# Predict Test ###############

#Config
CONFIG = {
    "seed": 24,
    "img_size": 384,
    "model_name": "hf_hub:timm/efficientnet_b3.ra2_in1k",
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}



#redefine ISICDataset to read from hdf
class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target,
        }
    
 
#Create model
#BEST_WEIGHT = "/kaggle/input/isic-pytorch-training-baseline-image-only/AUROC0.5171_Loss0.3476_epoch35.bin"
BEST_WEIGHT = r'G:\\kaggle\isic-2024-challenge\efficientnet_b3\pytorch\AUROC0.5180_Loss0.4542_epoch11.bin'
model = ISICModel(CONFIG['model_name'], pretrained=False)
model.load_state_dict( torch.load(BEST_WEIGHT) )
model.to(CONFIG['device']);


#redin csv
test = pd.read_csv(PATH + 'test-metadata.csv', low_memory=False)
test['target'] = 0

sub =  pd.read_csv(PATH + 'sample_submission.csv', low_memory=False)

#Dataloaders
TEST_HDF = r'G:\\kaggle\isic-2024-challenge\test-image.hdf5'
test_dataset = ISICDataset(test, TEST_HDF, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], 
                          num_workers=0, shuffle=False, pin_memory=True)
#Predicting
preds = []
with torch.no_grad():
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:        
        images = data['image'].to(CONFIG["device"], dtype=torch.float)        
        batch_size = images.size(0)
        outputs = model(images)
        preds.append( outputs.detach().cpu().numpy() )
preds = np.concatenate(preds).flatten()


test["target"] = preds
#test.to_csv("submission.csv", index=False)


##############################################
################# Predict Train ##############