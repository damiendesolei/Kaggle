# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:40:04 2025

@author: zrj-desktop
"""


import autograd
import autograd_gamma
import interface_meta
import formulaic
import lifelines


import pytorch_lightning
import sklearn
import torchmetrics
#import pytorch_tabnet
#import einops
#import pytorch_tabular


from pathlib import Path
#from metric import score
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

ROOT_DATA_PATH = r'G:\\kaggle\equity-post-HCT-survival-predictions\\'

pd.set_option('display.max_columns', 100)

train = pd.read_csv(ROOT_DATA_PATH+"train.csv")
test = pd.read_csv(ROOT_DATA_PATH+"test.csv")

CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail', 

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match',

    # Medication/operation related data
    'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 'conditioning_intensity'
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',
    
    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]

OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


def preprocess_data(df):
    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
    df[OTHER_NUMERICAL_VARIABLES] = df[OTHER_NUMERICAL_VARIABLES].fillna(df[OTHER_NUMERICAL_VARIABLES].median())

    return df

train = preprocess_data(train)
test = preprocess_data(test)


def features_engineering(df):
    # Change year_hct to relative year from 2000
    df['year_hct'] = df['year_hct'] - 2000
    
    return df


train = features_engineering(train)
test = features_engineering(test)

train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')

FEATURES = train.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()





import xgboost as xgb
print("Using XGBoost version",xgb.__version__)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier



# FOLDS = 5
# kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# oof_xgb = np.zeros(len(train))
# pred_efs = np.zeros(len(test))

# for i, (train_index, test_index) in enumerate(kf.split(train, train["efs"])):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train.loc[train_index, FEATURES].copy()
#     y_train = train.loc[train_index, "efs"]
#     x_valid = train.loc[test_index, FEATURES].copy()
#     y_valid = train.loc[test_index, "efs"]
#     x_test = test[FEATURES].copy()

#     model_xgb = XGBClassifier(
#         device="cuda",
#         max_depth=3,  
#         colsample_bytree=0.7129400756425178, 
#         subsample=0.8185881823156917, 
#         n_estimators=20_000, 
#         learning_rate=0.04425768131771064,  
#         eval_metric="auc", 
#         early_stopping_rounds=50, 
#         objective='binary:logistic',
#         scale_pos_weight=1.5379160847615545,  
#         min_child_weight=4,
#         enable_categorical=True,
#         gamma=3.1330719334577584
#     )
#     model_xgb.fit(
#         x_train, y_train,
#         eval_set=[(x_valid, y_valid)],  
#         verbose=100
#     )

#     # INFER OOF (Probabilities -> Binary)
#     oof_xgb[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
#     # INFER TEST (Probabilities -> Average Probs)
#     pred_efs += model_xgb.predict_proba(x_test)[:, 1]

# # COMPUTE AVERAGE TEST PREDS
# pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

# # EVALUATE PERFORMANCE
# accuracy = accuracy_score(train["efs"], oof_xgb)
# f1 = f1_score(train["efs"], oof_xgb)
# roc_auc = roc_auc_score(train["efs"], oof_xgb)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"ROC AUC Score: {roc_auc:.4f}")




# Define the parameter space
import optuna

def objective(trial):
    
    #n_estimators = trial.suggest_int('n_estimators', 500, 10000, step=500)
    n_estimators = 20_000
    param = {
        'objective': 'binary:logistic',  
        'eval_metric': 'auc', 
        'booster': 'gbtree',
        'device_type': 'cpu',  
        #'gpu_use_dp': True,

        #'n_estimators': 20_000,
        #'n_estimators': trial.suggest_int('n_estimators', 800, 2000, step=200),
        'max_depth': trial.suggest_int('max_depth', 12, 48, step=2),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
        'min_child_weight': trial.suggest_int('min_child_weight', 36, 512, step=4), 

        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        #'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1, log=True),

        'seed': 2025,
        'verbosity': 0,
        "disable_default_eval_metric": 1,  # Disable default eval metric logs
    }

    # Time series cross-validation
    skf = StratifiedKFold(n_splits=4, shuffle=False)
    scores = []
    
    for i, (train_index, test_index) in enumerate(skf.split(train, train["efs"])):       
        
        x_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, "efs"]
        x_valid = train.loc[test_index, FEATURES].copy()
        y_valid = train.loc[test_index, "efs"]
        #x_test = test[FEATURES].copy()
               
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)

        # Train XGBoost model
        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=0
        )
        
        # Predict on validation set
        y_pred = (model.predict(dvalid) > 0.5).astype(int)
    
        auc = roc_auc_score(y_valid, y_pred)  # WMAE for regression
        scores.append(auc)  
    
    mean_auc = np.mean(scores)
    
    return mean_auc



# Run Optuna study
N_HOUR = 2
CORES = 6

print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=3600*N_HOUR, n_jobs=CORES)  # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best mae:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
OUT_PATH = r'G:\\kaggle\equity-post-HCT-survival-predictions\models\\'
file_name = f"Xgboost_for_NN_mask_auc_{best_score:.6f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(OUT_PATH+file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")