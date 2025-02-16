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
model_path = r'G:\\kaggle\hackathon-ia-prevision-du-commerce-local\model\\'



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
model_name = f'glm_cv_{len(features)}_parameters'






X = train[features]
y = train['y_value']


# Initialize scaler
scaler = StandardScaler()

# Fit and transform the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# Ensure X has an intercept term for statsmodels
X = sm.add_constant(X)


# Define splits and tuning limit
N_SPLITS = 5
#SPLIT_LENGTH = timedelta(weeks=6)
HOURS = 1
CORES = 4

# Define Optuna objective function
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1, log=True)  # Alpha search space
    tscv = TimeSeriesSplit(n_splits=5)  # Time series cross-validation
    #tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=int(SPLIT_LENGTH.total_seconds()/(24*60*60)))
    
    mae_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        #w_train, w_test = w.iloc[train_idx], w.iloc[test_idx]  # Use same split for weights

        # Fit Lasso-regularized GLM
        model = GLM(y_train, X_train, family=Gamma(link=Log()))
        
        try:
            result = model.fit_regularized(alpha=alpha, L1_wt=1.0, method='elastic_net')
    
            # Predict using log-link transformation
            y_pred = result.predict(X_test)  
    
            # Compute weighted MAE
            mae = mean_absolute_error(y_test, y_pred)

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
print(f"Best alpha: {best_alpha}")
#Trial 1 finished with value: 72.24307522090697 and parameters: {'alpha': 0.00071874820494885}



# Fit final model with best alpha
final_model = GLM(y, X, family=Gamma(link=Log())).fit_regularized(alpha=best_alpha, L1_wt=1.0, method='elastic_net') #Lasso


print("Final model coefficients:", final_model.params)
final_coefficient = pd.DataFrame(final_model.params)
final_coefficient.rename(columns={0: 'coefficient'}, inplace=True)
final_coefficient.to_csv(f'{model_name}_coefficients_{best_alpha}.csv')

# Load test data
X_test = test[features] 

# Fit and transform the DataFrame
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

# Ensure X has an intercept term for statsmodels
X_test = sm.add_constant(X_test)

# Predict
test_predictions = final_model.predict(X_test)

print(test_predictions)

test_pred = pd.DataFrame(test_predictions)
test_pred

# Crete the submission
submission = test_pred
submission.rename(columns={0: 'y_value'}, inplace=True)
submission


submission.to_csv(f"{model_name}_submission_{best_alpha}.csv",index=True)