# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:52:43 2024

@author: zrj-desktop
"""

import numpy as np
import pandas as pd
#import pandas.api.types

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import lightgbm as lgb

import re




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

#integer encode the tbp_lv_location
def extract_location(row):
    match1 = re.search(r'(Top|top)', str(row['tbp_lv_location']))   
    match2 = re.search(r'(Middle|middle)', str(row['tbp_lv_location']))
    match3 = re.search(r'(Bottom|bottom)', str(row['tbp_lv_location']))
    match4 = re.search(r'(Upper|upper)', str(row['tbp_lv_location']))
    match5 = re.search(r'(Lower|lower)', str(row['tbp_lv_location']))
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
train[['tbp_lv_location','location']].drop_duplicates() #show mapping



##############################################
######### feature probing - EDA ##############
sns.kdeplot(data=train, x='sex_IntEncoded', hue='target')
sns.countplot(data=train, x='target', hue='sex')
sns.countplot(data=train[train.target==1], x='target', hue='sex') #For target=1, 'male' has a higher number

sns.kdeplot(data=train, x='tbp_tile_type_IntEncoded', hue='target')
sns.countplot(data=train, x='target', hue='tbp_tile_type')
sns.countplot(data=train[train.target==0], x='target', hue='tbp_tile_type') #For target==0, '3D: XP' has a higher number

sns.kdeplot(data=train, x='Previously_Insured', hue='Response');
sns.countplot(data=train, x='target', hue='tbp_lv_location')
sns.countplot(data=train[train.target==1], x='target', hue='tbp_lv_location')

sns.kdeplot(data=train, x='Policy_Sales_Channel', hue='Response');
sns.countplot(data=train, x='target', hue='location')

sns.kdeplot(data=train, x='Vintage', hue='Response');
sns.countplot(data=train, x='Response', hue='Vintage')

sns.kdeplot(data=train, x='Vehicle_Age_encoded', hue='Response');
sns.countplot(data=train, x='Response', hue='Vehicle_Age_encoded')


sns.kdeplot(data=train, x='Annual_Premium', hue='Response');
sns.kdeplot(data=train, x='Region_Code', hue='Response');

sns.kdeplot(data=train, x='Gender_encoded', hue='Response');
sns.countplot(data=train, x='Response', hue='Gender_encoded')

sns.kdeplot(data=train, x='Driving_License', hue='Response');
sns.kdeplot(data=train, x='random', hue='Response');
######### End of feature probing EDA #########
##############################################