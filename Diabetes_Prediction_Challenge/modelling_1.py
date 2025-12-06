# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 14:30:14 2025

@author: zrj-desktop
"""

import pandas as pd
import numpy as np
#import polars as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


import optuna
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

pd.set_option('display.max_columns', None)



PATH = 'G:/kaggle/Diabetes_Prediction_Challenge/playground-series-s5e12/'


train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')


#### Feature ####

#Gender
def encode_gender(df):
    df['gender_Other'] = df['gender'].apply(lambda x: 1 if x == 'Other' else 0)
    df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    return df


#ethnicity
def encode_ethnicity(df):
    df['ethnicity_Hispanic'] = df['ethnicity'].apply(lambda x: 1 if x == 'Hispanic' else 0)
    df['ethnicity_Black'] = df['ethnicity'].apply(lambda x: 1 if x == 'Black' else 0)
    df['ethnicity_Asian'] = df['ethnicity'].apply(lambda x: 1 if x == 'Asian' else 0)
    df['ethnicity_Other'] = df['ethnicity'].apply(lambda x: 1 if x == 'Other' else 0)
    return df


#education_level
def encode_education_level(df):
    df['education_level_Graduate'] = df['education_level'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['education_level_Postgraduate'] = df['education_level'].apply(lambda x: 1 if x == 'Postgraduate' else 0)
    df['education_level_No_formal'] = df['education_level'].apply(lambda x: 1 if x == 'No formal' else 0)
    return df


#income_level
def encode_income_level(df):
    df['income_level_Lower_Middle'] = df['income_level'].apply(lambda x: 1 if x == 'Lower-Middle' else 0)
    df['income_level_Upper_Middle'] = df['income_level'].apply(lambda x: 1 if x == 'Upper-Middle' else 0)
    df['income_level_Low'] = df['income_level'].apply(lambda x: 1 if x == 'Low' else 0)
    df['income_level_High'] = df['income_level'].apply(lambda x: 1 if x == 'High' else 0)
    return df


#smoking_status
def encode_smoking_status(df):
    df['smoking_status_Current'] = df['smoking_status'].apply(lambda x: 1 if x == 'Current' else 0)
    df['smoking_status_Former'] = df['smoking_status'].apply(lambda x: 1 if x == 'Former' else 0)
    return df



#### Process ####
train['label'] = 'train'
test['label'] = 'test'
df = pd.concat([train, test], ignore_index=True)

df = encode_gender(df) 
df = encode_ethnicity(df)
df = encode_education_level(df)
df = encode_income_level(df)
df = encode_smoking_status(df)




#### standardize ####
scaler = MinMaxScaler()  

col_to_scaled = [
    'age', 
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    ]

col_scaled = [
    'age_1', 
    'alcohol_consumption_per_week_1',
    'physical_activity_minutes_per_week_1',
    'diet_score_1',
    'sleep_hours_per_day_1',
    'screen_time_hours_per_day_1',
    'bmi_1',
    'systolic_bp_1',
    'diastolic_bp_1',
    'heart_rate_1',
    'cholesterol_total_1',
    'hdl_cholesterol_1',
    'ldl_cholesterol_1',
    'triglycerides_1',
    ]

df[col_scaled] = scaler.fit_transform(df[col_to_scaled])



#### Features ####
FEATURES = ['age_1', 'alcohol_consumption_per_week_1',
       'physical_activity_minutes_per_week_1', 'diet_score_1',
       'sleep_hours_per_day_1', 'screen_time_hours_per_day_1', 'bmi_1',
       'waist_to_hip_ratio', 'systolic_bp_1', 'diastolic_bp_1', 'heart_rate_1',
       'cholesterol_total_1', 'hdl_cholesterol_1', 'ldl_cholesterol_1',
       'triglycerides_1', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'gender_Other',
       'gender_Female', 'ethnicity_Hispanic', 'ethnicity_Black',
       'ethnicity_Asian', 'ethnicity_Other', 'education_level_Graduate',
       'education_level_Postgraduate', 'education_level_No_formal',
       'income_level_Lower_Middle', 'income_level_Upper_Middle',
       'income_level_Low', 'income_level_High', 'smoking_status_Current',
       'smoking_status_Former']



#### Create local valid ####
train = df[df.label=='train']
test = df[df.label=='test']

train, valid = train_test_split(train, test_size=0.15, stratify=train['diagnosed_diabetes'], random_state=2025)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
print('train shape:', train.shape)
print('valid shape:', valid.shape)


#### LASSO ####
l1_weight = 1

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
    global l1_weight
    
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(skf.split(train, train['diagnosed_diabetes'])):
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, 'diagnosed_diabetes']
        X_valid = train.loc[test_index, FEATURES].copy()
        y_valid = train.loc[test_index, 'diagnosed_diabetes']
        
        model = GLM(y_train, X_train, family=Binomial())

        
        try:
            result = model.fit_regularized(alpha=alpha, L1_wt=l1_weight, method='elastic_net', maxiter=10_000)
            # predict on holdout set
            y_pred = result.predict(X_valid)
            auc = roc_auc_score(y_valid, y_pred)
        except Exception as e:
            print(f'Failed for alpha={alpha}: {e}')
            return 0
        
        scores.append(auc)
    
    return np.mean(scores)


#### Run Hyper Parameter Tuning ####
HOURS = 1
CORES = 3
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=3600*HOURS, n_jobs=CORES)  # Run n HOURS

# Best alpha from optimization
best_alpha = study.best_params["alpha"]
best_mae = study.best_value
print(f"Best alpha: {best_alpha}")
print("Best AUC:", study.best_value)


#### Fit final model with best alpha ####
model_name = 'GLM'
final_model = GLM(train['diagnosed_diabetes'], train[FEATURES], family=Binomial()).fit_regularized(alpha=best_alpha, L1_wt=1.0, method='elastic_net') #Lasso


print("Final model coefficients:", final_model.params)
final_coefficient = pd.DataFrame(final_model.params)
final_coefficient.rename(columns={0: 'coefficient'}, inplace=True)
final_coefficient.to_csv(PATH+f'{model_name}_coefficients_{best_alpha}.csv')