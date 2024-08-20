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
from colorama import Fore, Style



##############################################
######### read in data #######################
PATH = r'G:\\kaggle\isic-2024-challenge\\'

train = pd.read_csv(PATH + 'train-metadata.csv', low_memory=False)
test = pd.read_csv(PATH + 'test-metadata.csv', low_memory=False)
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
sns.kdeplot(data=train, x='sex_IntEncoded', hue='target')
sns.countplot(data=train, x='target', hue='sex')
sns.countplot(data=train[train.target==1], x='target', hue='sex') #For target=1, 'male' has a higher number

sns.kdeplot(data=train, x='tbp_tile_type_IntEncoded', hue='target')
sns.countplot(data=train, x='target', hue='tbp_tile_type')
sns.countplot(data=train[train.target==0], x='target', hue='tbp_tile_type') #For target==0, '3D: XP' has a higher number

#sns.kdeplot(data=train, x='tbp_lv_location_Intencoded', hue='target');
sns.countplot(data=train, x='target', hue='tbp_lv_location')
sns.countplot(data=train[train.target==1], x='target', hue='tbp_lv_location')

sns.kdeplot(data=train, x='location_Intencoded', hue='target');
sns.countplot(data=train, x='target', hue='location')

sns.kdeplot(data=train, x='position_IntEncoded', hue='target');
sns.countplot(data=train, x='target', hue='position')

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

initial_features = num_cols + ['sex_IntEncoded', 'tbp_tile_type_IntEncoded', 'location_IntEncoded', 'tbp_lv_location_simple_IntEncoded', 'position_IntEncoded']


#lgb ~ 1m
lgb_model = lgb.LGBMClassifier(verbose=-1, eval_metric='custom_score', device='cpu')
cross_validate(lgb_model, 'LightGBM_Untuned', features=initial_features)
# Fold 0: tr_auc=0.01967, val_auc=0.01787
# Fold 1: tr_auc=0.02095, val_auc=0.02046
# Fold 2: tr_auc=0.03968, val_auc=0.02996
# Fold 3: tr_auc=0.02669, val_auc=0.02944
# Fold 4: tr_auc=0.02709, val_auc=0.02500
# Overall val=0.02455 LightGBM_Untuned   0 min
# LightGBM_Untuned Fitting started from 2024-08-20 14:41:16.252451


#1.lgb
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
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=3600)
#study_summaries = optuna.study.get_all_study_summaries()
# Trial 0 finished with value: 0.17058399728141566 and parameters: {'iterations': 1564, 'learning_rate': 0.021861424132643192, 'colsample_bylevel': 0.09839000386075163, 'depth': 15, 'min_data_in_leaf': 4169, 'subsample': 0.24265885764272066, 'colsample_bytree': 0.40469534371300453, 'max_bin': 15603, 'reg_lambda': 1.6371338817060606e-05, 'reg_alpha': 4.4221534988928415, 'pos_bagging_fraction': 0.5623107344062629, 'neg_bagging_fraction': 0.7514980166710389}. Best is trial 0 with value: 0.17058399728141566.
#Trial 21 finished with value: 0.1715800231544985 and parameters: {'iterations': 1401, 'learning_rate': 0.05032974645541588, 'colsample_bylevel': 0.02898675577640679, 'depth': 18, 'min_data_in_leaf': 17839, 'subsample': 0.8920051990627909, 'colsample_bytree': 0.6202745141075666, 'max_bin': 125990, 'reg_lambda': 0.002175710371681295, 'reg_alpha': 2.504363052953543e-05, 'pos_bagging_fraction': 0.14508814884263116, 'neg_bagging_fraction': 0.4948871177523545}. Best is trial 21 with value: 0.1715800231544985. 
plotly_config = {"staticPlot": True}
fig = plot_optimization_history(study)
fig.show(config=plotly_config)
fig = plot_param_importances(study)
fig.show(config=plotly_config)