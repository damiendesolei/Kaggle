# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:54:35 2025

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




from sklearn.preprocessing import TargetEncoder
print('<<< encoding series_name >>>')

# encode series_name + quarter
train['series_qtr'] = train['series_name'] + '_Q' + train['quarter'].astype(str)
test['series_qtr'] = test['series_name'] + '_Q' + test['quarter'].astype(str)

encoder = TargetEncoder(smooth=2, target_type='continuous')
encoder.fit(train[['series_qtr']], train['y_value'])

train['series_qtr_te'] = encoder.transform(train[['series_qtr']]).flatten()
test['series_qtr_te'] = encoder.transform(test[['series_qtr']]).flatten() #transform (not fit_transform)


# encode series_name + quarter
train['series_day'] = train['series_name'] + '_D' + train['day'].astype(str)
test['series_day'] = test['series_name'] + '_D' + test['day'].astype(str)

encoder = TargetEncoder(smooth=2, target_type='continuous')
encoder.fit(train[['series_day']], train['y_value'])

train['series_day_te'] = encoder.transform(train[['series_day']]).flatten()
test['series_day_te'] = encoder.transform(test[['series_day']]).flatten() #transform (not fit_transform)






# intial features with only numeric columns
features_0 = [col for col in test.columns if (test[col].dtype != 'object' and test[col].dtype != 'datetime64[ns]')]

# check column difference
print(f'{np.setdiff1d(train.columns, test.columns)} in train, but not in test')
features_not_in_test = np.setdiff1d(train.columns, test.columns) 

remove_features = ['series_name'] + features_not_in_test

features = [feature for feature in features_0 if feature not in remove_features]

# Setup model name to tune and predict
model_name = f'lgb_cv_{len(features)}_parameters'






# Use all train for hyper parameter tuning
X_train = train[features]
y_train = train['y_value']



# Define TimeSeriesSplit parameters
N_SPLITS = 5
#SPLIT_LENGTH = timedelta(weeks=4)  # Test size of 2 weeks as per requirement
N_HOURS = 1

# Define the parameter space
def objective(trial):
    param = {
        'objective': 'regression',  
        'metric': 'mae',  
        'boosting_type': 'gbdt',
        'device_type': 'cpu', 
        'gpu_use_dp': True,
        
        'linear_tree': True,
        
        'n_estimators': trial.suggest_int('n_estimators', 300, 700, step=100),
        'max_depth': trial.suggest_int('max_depth', 2, 16, step=1),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  
        'num_leaves': trial.suggest_int('num_leaves', 12, 128, step=1), 
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),  
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),  
        #'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.001, 0.1),
        
        'verbose': -1,
        'seed' : 2025

    }

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        #w_train_fold, w_val_fold = w_train.iloc[train_idx], y_train.iloc[val_idx]
    
        # Create a LightGBM dataset
        dtrain = lgb.Dataset(X_train_fold, y_train_fold)
        dval = lgb.Dataset(X_val_fold, y_val_fold, reference=dtrain)

        # Train LightGBM model
        model = lgb.train(
            params=param,
            train_set=dtrain,
            valid_sets=[dval],
            #feval=lambda y_pred, dval: r2_lgb(dval.get_label(), y_pred, dval.get_weight()),  # Use weights in the custom metric
            callbacks=[lgb.early_stopping(100)]
        )

        # Predict on validation set
        y_pred = model.predict(X_val_fold)
    
        mae = mean_absolute_error(y_val_fold, y_pred)  # WMAE for regression
        scores.append(mae)  
    
    mean_mae = np.mean(scores)
    
    return mean_mae


# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=3600*N_HOURS, n_jobs=4) # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best wmae:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
file_name = model_path + model_name + f"_mae_{best_score:.4f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")



# Best mae: 15.466090858184723
STUDY=True
if not STUDY:
    best_params = {
        'objective': 'regression',  
        'metric': 'mae',  
        'boosting_type': 'gbdt',
        'device_type': 'cpu', 
        
        #'linear_tree': True,
        
         'n_estimators': 400,
         'max_depth': 2,
         'learning_rate': 0.08692607383010294,
         'num_leaves': 75,
         'feature_fraction': 0.847425228674351,
         'bagging_fraction': 0.8175683301880504,
         'lambda_l1': 0.008422365379531225,
         'lambda_l2': 0.003503958237043179
    }
    
    
    
    
# Load the training data
X = X_train
y = y_train


# Initialize StratifiedKFold for cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


# Initialize variables for OOF predictions, test predictions, and feature importances
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))  # Ensure 'test' DataFrame is loaded
feature_importances = []
models = []
valid_maes = []



# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(tscv.split(X, y)):
    print(f"Training fold {fold + 1}")
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    

    # Initialize and train the model
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train_fold, 
              y_train_fold, 
              eval_metric='mae'
              #callbacks=[lgb.early_stopping(100)]
             )

    # Generate validation predictions
    y_pred_valid = model.predict(X_valid_fold)
    fold_mae = mean_absolute_error(y_valid_fold, y_pred_valid)
    valid_maes.append(fold_mae)
    oof_predictions[valid_idx] = y_pred_valid

    # Generate test predictions and accumulate
    test_pred = model.predict(test[features])
    test_predictions += test_pred / tscv.n_splits

    # Save model and feature importance
    joblib.dump(model, f"{model_path}{model_name}_fold{fold+1}_auc_{fold_mae:.5f}.model")
    models.append(model)
    
    # Collect feature importances
    fold_importance = pd.DataFrame({
        'Feature': model.feature_name_,
        'Importance': model.feature_importances_,
        'Fold': fold + 1
    })
    feature_importances.append(fold_importance)

    print(f"Fold {fold + 1} AUC: {fold_mae:.5f}")
    
    
    
# Calculate overall metrics
overall_mae = mean_absolute_error(y, oof_predictions)
print(f"Average Validation MAE: {np.mean(valid_maes):.5f}")
print(f"Overall OOF MAE: {overall_mae:.5f}")


# Save OOF predictions and true values
oof_df = X.copy()
oof_df['y_value'] = y
oof_df['pred'] = oof_predictions
oof_df.to_csv(f"{model_path}{model_name}_oof_predictions_mae_{overall_mae:.5f}.csv", index=False)


# Aggregate and save feature importances
feature_importances_df = pd.concat(feature_importances)
average_importance = feature_importances_df.groupby('Feature')['Importance'].mean().reset_index()
average_importance = average_importance.sort_values('Importance', ascending=False)
average_importance.to_csv(f"{model_path}{model_name}_avg_feature_importance.csv", index=False)


# Plot average feature importance
plt.figure(figsize=(12, 11))
sns.barplot(x='Importance', y='Feature', data=average_importance.head(55))
plt.title('Top Features (Average Importance)')
plt.tight_layout()
#plt.savefig(f"{model_path}{model_name}_average_feature_importance.png")
plt.show()


# CHeck predictions
test_pred = pd.DataFrame(test_predictions)
#test_pred                                                                                                                 


# Crete the submission
submission = pd.DataFrame()
submission['ID'] = test.index
submission['y_value'] = test_pred
#submission.rename(columns={0: 'y_value'}, inplace=True)
submission.to_csv(f"{model_path}{model_name}_submission_mae_{np.mean(valid_maes):.5f}.csv",index=False)


# For submission notes
print(f"Best hyper mae: {best_score}")
print(f"Average Validation MAE: {np.mean(valid_maes):.5f}")
print(best_params)