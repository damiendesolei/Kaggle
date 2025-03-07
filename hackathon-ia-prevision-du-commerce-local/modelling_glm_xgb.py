# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:40:21 2025

@author: zrj-desktop
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# GLM
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log

# 
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import optuna

#
from datetime import datetime, timedelta

#
from tqdm import tqdm, tqdm_notebook
import joblib
from joblib import Parallel, delayed

#
import matplotlib.pyplot as plt
import seaborn as sns

#
import warnings
warnings.simplefilter("ignore")  # Suppress all warnings





PATH = r'G:\\kaggle\hackathon-ia-prevision-du-commerce-local\\'
MODEL_PATH = r'G:\\kaggle\hackathon-ia-prevision-du-commerce-local\model\\'



train = pd.read_csv(PATH+'train.csv', parse_dates=['date'], index_col=['ID'])
print("Train shape",train.shape)
train = train[train.y_value>=0] # remove negative rows
train = train[train.date>='2023-01-01 00:00:00'] # only use recent data
print("Train shape post filter",train.shape)
test = pd.read_csv(PATH+'test.csv', parse_dates=['date'], index_col=['ID'])
print("Test shape",train.shape)




train.groupby(['series_name']).agg({'y_value': 'mean'})



def category_features(df):
    print('<<< extract categorical features >>>')
    
    df['is_100k'] = df['series_name'].str.contains('100k', na=False).astype(int)
    df['series_name'] = df['series_name'].str.replace(r'_base.*', '', regex=True)

    return df
    
train = category_features(train)
test = category_features(test)



def date_features(df):
    print('<<< create date features >>>')
    
    df['year'] = df['date'].dt.year

    df['quarter']=df['date'].dt.quarter
    df['sin_quarter']=np.sin(2*np.pi*df['quarter']/4)
    df['cos_quarter']=np.cos(2*np.pi*df['quarter']/4)

    df['month']=df['date'].dt.month
    df['sin_month']=np.sin(2*np.pi*df['month']/12)
    df['cos_month']=np.cos(2*np.pi*df['month']/12)

    df['day']=df['date'].dt.day
    df['sin_day']=np.sin(2*np.pi*df['day']/30)
    df['cos_day']=np.cos(2*np.pi*df['day']/30)

    df['day_of_week']=df['date'].dt.dayofweek

    return df

train = date_features(train)
test = date_features(test)


# One-hot day of week (weekends tend to have higher y)


# Add Fourier features for seasonality
reference_date = pd.Timestamp('2000-01-01')  # UNIX Epoch
train['days_since_2000'] = (train['date'] - reference_date).dt.days
test['days_since_2000'] = (test['date'] - reference_date).dt.days

def add_fourier_terms(df, period, order):
    for i in range(1, order + 1):
        df[f'sin_year_{i}'] = np.sin(2 * np.pi * i * df['days_since_2000'] / period)
        df[f'cos_year_{i}'] = np.cos(2 * np.pi * i * df['days_since_2000'] / period)
    return df

train = add_fourier_terms(train, period=365, order=3)
test = add_fourier_terms(test, period=365, order=3)




#from sklearn.preprocessing import TargetEncoder
#print('<<< encoding series_name >>>')

train['series_qtr'] = train['series_name'] + '_Q' + train['quarter'].astype(str)
test['series_qtr'] = test['series_name'] + '_Q' + test['quarter'].astype(str)


print("<<< one hot encoding series_name >>>")
train = pd.get_dummies(train, columns=['series_qtr'], dtype=int)
test = pd.get_dummies(test, columns=['series_qtr'], dtype=int)
# set up smoothing


# Drop and columns that contains Q1, Q2 and Q3 - since test do not contain them
col_2_drop = [item for item in train.columns if any(keyword in item for keyword in ["Q1", "Q2", "Q3"])]
train = train.drop(columns=col_2_drop)

# encoder = TargetEncoder(smooth=2, target_type='continuous')
# encoder.fit(train[['series_qtr']], train['y_value'])

# # encode series_name
# train['series_qtr_te'] = encoder.transform(train[['series_qtr']]).flatten()
# test['series_qtr_te'] = encoder.transform(test[['series_qtr']]).flatten() #transform (not fit_transform)


# check column difference
print(f'{np.setdiff1d(train.columns, test.columns)} in train, but not in test')
features_not_in_test = np.setdiff1d(train.columns, test.columns) 


# intial features with only numeric columns
features_0 = [col for col in test.columns if (test[col].dtype != 'object' and test[col].dtype != 'datetime64[ns]')]

remove_features = ['series_name'] + features_not_in_test

features = [feature for feature in features_0 if feature not in remove_features]

# Setup model name to tune and predict
model_name = f'glm_{len(features)}_parameters'






X = train[features]
y = train['y_value']


# Initialize scaler
scaler = StandardScaler()

# Fit and transform the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# Ensure X has an intercept term for statsmodels
X = sm.add_constant(X)


# Define splits and tuning limit
#N_SPLITS = 5
#SPLIT_LENGTH = timedelta(weeks=6)

# Use Nov 2024 for test -> 1 fold
train_idx = train.query('date<="2024-10-31"').index.to_numpy() 
valid_idx = train.query('date>"2024-10-31"').index.to_numpy() 




STUDY_GLM = False
HOURS = 0.5
CORES = 1
if STUDY_GLM:
    # Define Optuna objective function
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1, log=True)  # Alpha search space
        #tscv = TimeSeriesSplit(n_splits=5)  # Time series cross-validation
       
        mae_scores = []
    
    #for train_idx, test_idx in zip(train_index, test_index):#tscv.split(X):
        X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        #w_train, w_test = w.iloc[train_idx], w.iloc[test_idx]  # Use same split for weights
    
        # Fit Lasso-regularized GLM
        model = GLM(y_train, X_train, family=Gamma(link=Log()))
        
        try:
            result = model.fit_regularized(alpha=alpha, L1_wt=1.0, method='elastic_net')
    
            # Predict on the validation set
            y_pred = result.predict(X_valid)  
    
            # Compute MAE
            mae = mean_absolute_error(y_valid, y_pred)
    
        except Exception as e:
            print(f"Failed for alpha={alpha}: {e}")
            return np.inf  # Penalize Optuna for bad trials
            
        mae_scores.append(mae)
    
        return np.mean(mae_scores)  # Minimize average WMAE
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")  # Minimize WMAE
    study.optimize(objective, timeout=3600*HOURS, n_jobs=CORES)  # Run n HOURS
    
    # Best alpha from optimization
    best_alpha = study.best_params["alpha"]
    best_mae = study.best_value
    print(f"Best alpha: {best_alpha}")
    print("Best mae:", study.best_value)
    #Trial 1 finished with value: 72.24307522090697 and parameters: {'alpha': 0.00071874820494885}

if not STUDY_GLM:
    best_alpha = 0.04835298884070995


# Fit final model with best alpha
final_model = GLM(y.loc[train_idx], X.loc[train_idx], family=Gamma(link=Log())).fit_regularized(alpha=best_alpha, L1_wt=1.0, method='elastic_net') #Lasso

if STUDY_GLM:
    print("Final model coefficients:", final_model.params)
    final_coefficient = pd.DataFrame(final_model.params)
    final_coefficient.rename(columns={0: 'coefficient'}, inplace=True)
    final_coefficient.to_csv(MODEL_PATH+f'{model_name}_coefficients_{best_alpha}.csv')



# Load test data
X_test = test[features] 

# Fit and transform the DataFrame
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

# Ensure X has an intercept term for statsmodels
X_test = sm.add_constant(X_test)

# Predict on test set
test_pred_glm = final_model.predict(X_test)
print(test_pred_glm)

test_pred_glm = pd.DataFrame(test_pred_glm)
test_pred_glm
                                 
# Get last month in train set as valid set
X_valid = X.loc[valid_idx]
y_valid = y.loc[valid_idx]
valid_pred_glm = final_model.predict(X_valid)





# Crete the submission
# submission = glm_test_pred
# submission.rename(columns={0: 'y_value'}, inplace=True)
# submission


#submission.to_csv(MODEL_PATH+f"{model_name}_submission_{best_mae}.csv",index=True)








#### Calculate the residual from GLM and fit Xgb ####
train_pred_glm = final_model.predict(X)
train['y_glm'] = train_pred_glm
y_residual = train['y_value'] - train_pred_glm



# load data
train = pd.read_csv(PATH+'train.csv', parse_dates=['date'], index_col=['ID'])
print("Train shape",train.shape)
train = train[train.y_value>=0] # remove negative rows
train = train[train.date>='2023-01-01 00:00:00'] # only use recent data
print("Train shape post filter",train.shape)
test = pd.read_csv(PATH+'test.csv', parse_dates=['date'], index_col=['ID'])
print("Test shape",train.shape)


def category_features(df):
    print('<<< extract categorical features >>>')
    
    df['is_100k'] = df['series_name'].str.contains('100k', na=False).astype(int)
    df['series_name'] = df['series_name'].str.replace(r'_base.*', '', regex=True)

    return df
    
train = category_features(train)
test = category_features(test)



def date_features(df):
    print('<<< create date features >>>')
    
    df['year'] = df['date'].dt.year

    df['quarter']=df['date'].dt.quarter
    df['sin_quarter']=np.sin(2*np.pi*df['quarter']/4)
    df['cos_quarter']=np.cos(2*np.pi*df['quarter']/4)

    df['month']=df['date'].dt.month
    df['sin_month']=np.sin(2*np.pi*df['month']/12)
    df['cos_month']=np.cos(2*np.pi*df['month']/12)

    df['day']=df['date'].dt.day
    df['sin_day']=np.sin(2*np.pi*df['day']/30)
    df['cos_day']=np.cos(2*np.pi*df['day']/30)

    df['day_of_week']=df['date'].dt.dayofweek

    return df

train = date_features(train)
test = date_features(test)



# Add Fourier features for seasonality
reference_date = pd.Timestamp('2000-01-01')  # UNIX Epoch
train['days_since_2000'] = (train['date'] - reference_date).dt.days
test['days_since_2000'] = (test['date'] - reference_date).dt.days

def add_fourier_terms(df, period, order):
    for i in range(1, order + 1):
        df[f'sin_year_{i}'] = np.sin(2 * np.pi * i * df['days_since_2000'] / period)
        df[f'cos_year_{i}'] = np.cos(2 * np.pi * i * df['days_since_2000'] / period)
    return df

train = add_fourier_terms(train, period=365, order=3)
test = add_fourier_terms(test, period=365, order=3)




#from sklearn.preprocessing import TargetEncoder
#print('<<< encoding series_name >>>')

train['series_qtr'] = train['series_name'] + '_Q' + train['quarter'].astype(str)
test['series_qtr'] = test['series_name'] + '_Q' + test['quarter'].astype(str)


# print("<<< one hot encoding series_name >>>")
# train = pd.get_dummies(train, columns=['series_qtr'], dtype=int)
# test = pd.get_dummies(test, columns=['series_qtr'], dtype=int)
# set up smoothing


# Drop and columns that contains Q1, Q2 and Q3 - since test do not contain them
# col_2_drop = [item for item in train.columns if any(keyword in item for keyword in ["Q1", "Q2", "Q3"])]
# train = train.drop(columns=col_2_drop)

# encoder = TargetEncoder(smooth=2, target_type='continuous')
# encoder.fit(train[['series_qtr']], train['y_value'])

# # encode series_name
# train['series_qtr_te'] = encoder.transform(train[['series_qtr']]).flatten()
# test['series_qtr_te'] = encoder.transform(test[['series_qtr']]).flatten() #transform (not fit_transform)


# check column difference
print(f'{np.setdiff1d(train.columns, test.columns)} in train, but not in test')
features_not_in_test = np.setdiff1d(train.columns, test.columns) 

CATEGORICAL_VARIABLES = ['series_name','series_qtr']
train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')

# intial features with only numeric columns
FEATURES = ['series_name', 'is_100k', 'year', 'quarter',
       'sin_quarter', 'cos_quarter', 'month', 'sin_month', 'cos_month', 'day',
       'sin_day', 'cos_day', 'day_of_week', 'days_since_2000', 'sin_year_1',
       'cos_year_1', 'sin_year_2', 'cos_year_2', 'sin_year_3', 'cos_year_3',
       'series_qtr'
]





HOURS = 0.5
CORES = 6
# Define the parameter space
import optuna

def objective(trial):
    
    #n_estimators = trial.suggest_int('n_estimators', 8000, 10000, step=200)
    n_estimators =10_000
    param = {
        'objective': 'reg:absoluteerror',  
        'eval_metric': 'mae', 
        'booster': 'gbtree',
        'device_type': 'cpu',  
        #'gpu_use_dp': True,

        'max_depth': trial.suggest_int('max_depth', 12, 48, step=2),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 512, step=2), 

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
    # skf = StratifiedKFold(n_splits=4, shuffle=False)
    scores = []
    
    #for i, (train_index, test_index) in enumerate(skf.split(train, train["efs"])):       
        
    x_train = train.loc[train_idx, FEATURES].copy()
    y_train = y_residual.loc[train_idx]
    x_valid = train.loc[valid_idx, FEATURES].copy()
    y_valid = y_residual.loc[valid_idx]
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
    y_pred = model.predict(dvalid)

    mae = mean_absolute_error(y_valid, y_pred)  # WMAE for regression
    scores.append(mae)  
    
    mean_mae = np.mean(scores)
    
    return mean_mae


print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=3600*HOURS, n_jobs=CORES)  # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best mae:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
file_name = f"Xgb_residual_params_mae_{best_score:.6f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(MODEL_PATH+file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {MODEL_PATH+file_name}")






#### fit the residual ####
from xgboost import XGBRegressor
print("Using XGBoost version",xgb.__version__)


pred_xgb = np.zeros(len(test))
#oof_xgb = np.zeros(len(train))

    
# Split the train and test
x_train = train.loc[train_idx, FEATURES].copy()
y_train = y_residual.loc[train_idx]
x_valid = train.loc[valid_idx, FEATURES].copy()
y_valid = y_residual.loc[valid_idx]
x_test = test[FEATURES].copy()


model_xgb = XGBRegressor(
    #device="cuda",
    n_estimators=10_000,
    max_depth=28,    
    learning_rate=0.0791527537157908,  
    min_child_weight=54,
    colsample_bytree=0.960133035668709,  
    subsample=0.923826282567079,
    reg_alpha=0.00132492748544914,
    reg_lambda=0.0376478752502566,
    early_stopping_rounds=100,
    enable_categorical=True,
)
model_xgb.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],  
    verbose=500 
)

# INFER OOF (Valid)
valid_pred_xgb = model_xgb.predict(x_valid)
# INFER TEST
test_pred_xgb = model_xgb.predict(x_test)

# Compute mae on the valid set 
valid_pred_glm_ = valid_pred_glm.to_numpy()
valid_pred = valid_pred_glm_ + valid_pred_xgb
local_mae = mean_absolute_error(y.loc[valid_idx], valid_pred)
print(local_mae)

#combine the glm pred and xgb residual
test_pred = np.zeros(test.shape[0])
test_pred_glm_ = test_pred_glm.to_numpy().reshape(-1,)
test_pred = test_pred_glm_ + test_pred_xgb


#Creat the submission file
sub = pd.read_csv(PATH+'test.csv')
sub['y_value'] = test_pred
sub[['ID','y_value']].to_csv(MODEL_PATH+f"GLM_Gamma_Log_Xgb_residual_submission_{local_mae}.csv",index=False)

print("Sub shape:",sub.shape)
sub.head()