# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:59:31 2025

@author: zrj-desktop
"""
#import warnings
#warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost.callback import EarlyStopping

import optuna
import joblib 
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

#https://www.kaggle.com/code/masayakawamata/s5e12-eda-xgb-competition-starter#4.2.-Robust-Target-Encoder-with-Internal-CV 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique', 
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the 
        global mean. If 'auto', an empirical Bayes estimate is used.
        
    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series).fillna(self.global_stats_[agg_func])
        
        if self.drop_original:
            X_transformed = X_transformed.drop(columns=self.cols_to_encode)
            
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)
                    
                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed = X_transformed.drop(columns=self.cols_to_encode)
            
        return X_transformed











PATH = 'G:/kaggle/Diabetes_Prediction_Challenge/playground-series-s5e12/'


train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')






#### Feature ####

#Gender
# def encode_gender(df):
#     temp = df.copy()
    
#     dic ={'Female': 0,
#           'Male': 1,
#           'Other': 2
#           }
#     temp['gender'] = temp['gender'].map(dic).fillna(2) 
        
#     return temp

#Gender
def encode_gender(df):
    df['gender_Other'] = df['gender'].apply(lambda x: 1 if x == 'Other' else 0)
    df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
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


#employment_status
def encode_employment_status(df):
    df['employment_status_Retired'] = df['employment_status'].apply(lambda x: 1 if x == 'Retired' else 0)
    df['employment_status_Unemployed'] = df['employment_status'].apply(lambda x: 1 if x == 'Unemployed' else 0)
    df['employment_status_Student'] = df['employment_status'].apply(lambda x: 1 if x == 'Student' else 0)
    return df




#### Process ####
# train['label'] = 'train'
# test['label'] = 'test'
# df = pd.concat([train, test], ignore_index=True)

# df = encode_gender(df) 
# df = encode_ethnicity(df)
# df = encode_education_level(df)
# df = encode_income_level(df)
# df = encode_smoking_status(df)
# df = encode_employment_status(df)

TE_COLS = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

#### Reduce RAM ####
import gc
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df





#### FEATRURES ####
FEATURES = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 
       # 'gender_Other',
       # 'gender_Female', 'ethnicity_Hispanic', 'ethnicity_Black',
       # 'ethnicity_Asian', 'ethnicity_Other', 'education_level_Graduate',
       # 'education_level_Postgraduate', 'education_level_No_formal',
       # 'income_level_Lower_Middle', 'income_level_Upper_Middle',
       # 'income_level_Low', 'income_level_High', 'smoking_status_Current',
       # 'smoking_status_Former',
       # 'employment_status_Retired',
       # 'employment_status_Unemployed', 'employment_status_Student'
       'TE_gender_mean', 'TE_ethnicity_mean', 'TE_income_level_mean', 
       'TE_smoking_status_mean', 'TE_employment_status_mean'
]





#### Create local valid ####
# train = df[df.label=='train']
# test = df[df.label=='test']

tr, val = train_test_split(train, test_size=0.10, stratify=train['diagnosed_diabetes'], random_state=2025)
tr = tr.reset_index(drop=True)
val = val.reset_index(drop=True)
print('tr shape:', tr.shape)
print('val shape:', val.shape)



#### Hyper Param ####
STUDY = True
N_HOUR = 3

if STUDY:
    # Hyper parameter tuning
    def objective(trial):
        # Define hyperparameters
        
        n_estimators = 500
        #n_estimators = trial.suggest_int('n_estimators', 100, 500, step=100)
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            #'tree_method': 'exact',
            'tree_method': 'gpu_hist',
            #'grow_policy': 'lossguide',
            
            'max_depth': trial.suggest_int('max_depth', 2, 24, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            #'max_leaves': trial.suggest_int('max_leaves', 2, 128, step=1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'alpha': trial.suggest_float("alpha", 0.001, 0.1, log=True),
            'lambda': trial.suggest_float("lambda", 0.001, 0.1, log=True),
            
            'seed': 2025,
            'verbosity': 0,
            "disable_default_eval_metric": 1,  # Disable default eval metric logs
            'device': 'cuda'
        }
    
        # Set up K-Fold cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2025)
        scores = []
    
        for train_idx, val_idx in skf.split(tr, tr['diagnosed_diabetes']):
            # Split data into training and validation sets
            X_train_fold, X_val_fold = tr.iloc[train_idx], tr.iloc[val_idx]
            y_train_fold, y_val_fold = tr['diagnosed_diabetes'].iloc[train_idx], tr['diagnosed_diabetes'].iloc[val_idx]
            
            # Target encode within each fold
            if len(TE_COLS) > 0:
                TE = TargetEncoder(cols_to_encode=TE_COLS, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
                X_train_fold = TE.fit_transform(X_train_fold, y_train_fold)
                X_val_fold = TE.transform(X_val_fold)
            
            # Create XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train_fold[FEATURES], label=y_train_fold)
            dvalid = xgb.DMatrix(X_val_fold[FEATURES], label=y_val_fold)
            
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
    
            # Calculate AUC
            auc = roc_auc_score(y_val_fold, y_pred)
            scores.append(auc)    
    
        mean_auc = np.mean(scores)
        return mean_auc
    
    # Run Optuna study
    print("Start running hyper parameter tuning..")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=3600*N_HOUR, n_jobs=3)  # 3600*n hour
    
    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best AUC:", study.best_value)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Format the file name with the best score
    file_name = PATH+'Xgb/'+f"xgb_auc_{best_score:.4f}.csv"
    
    # Save the best parameters to a CSV file
    df_param = pd.DataFrame([best_params])  # Convert to DataFrame
    df_param.to_csv(file_name, index=False)  # Save to CSV
    
    print(f"Best parameters saved to {file_name}")
    
    


#### Fit on full data ####
if not STUDY:
    xgb_params = {
        'n_estimators': 500, 
        'max_depth': 5, 
        'learning_rate': 0.09596056849177695, 
        'colsample_bytree': 0.6, 
        'subsample': 1, 
        'alpha': 0.008025776130982414, 
        'lambda': 0.01707429416017424,
        #'grow_policy': 'lossguide',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'random_state': 2025,
        'verbosity': 0,
        #'use_label_encoder': False,
        'early_stopping_rounds': 100
    }

# Load the training data
X = train#[FEATURES]
y = train['diagnosed_diabetes']

# skf
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

# Initialize variables for OOF predictions, test predictions, and feature importances
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))  # Ensure 'test' DataFrame is loaded
feature_importances = []
models = []
valid_aucs = []

# Convert best_params to XGBoost classifier parameters
if STUDY:
    xgb_params = {
        'n_estimators': 500, #best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        #'max_leaves': best_params['max_leaves'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample'],
        'reg_alpha': best_params['alpha'],
        'reg_lambda': best_params['lambda'],
        'tree_method': 'gpu_hist',
        #'grow_policy': 'lossguide',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 2025,
        'verbosity': 0,
        'use_label_encoder': False,
        'early_stopping_rounds': 100
    }

# cv loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}")
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    X_test_fold = test.copy() 
    
    # Target encode within each fold
    if len(TE_COLS) > 0:
        TE = TargetEncoder(cols_to_encode=TE_COLS, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
        X_train_fold = TE.fit_transform(X_train_fold, y_train_fold)
        X_valid_fold = TE.transform(X_valid_fold)
        X_test_fold = TE.transform(X_test_fold) # same encoding for test
    

    # Initialize and train the model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train_fold[FEATURES], y_train_fold,
        eval_set=[(X_valid_fold[FEATURES], y_valid_fold)],
        verbose=False
    )
    
    # Generate validation predictions
    y_pred_valid = model.predict_proba(X_valid_fold[FEATURES])[:, 1]  # Get probability of positive class
    fold_auc = roc_auc_score(y_valid_fold, y_pred_valid)
    valid_aucs.append(fold_auc)
    oof_predictions[valid_idx] = y_pred_valid

    # Generate test predictions and accumulate
    test_pred = model.predict_proba(X_test_fold[FEATURES])[:, 1]
    test_predictions += test_pred / skf.n_splits

    # Save model and feature importance
    joblib.dump(model, PATH+'Xgb/'+f"Xgboost_fold{fold+1}_auc_{fold_auc:.5f}.model")
    models.append(model)
    
    # Collect feature importances
    fold_importance = pd.DataFrame({
        'Feature': model.get_booster().feature_names,
        'Importance': model.feature_importances_,
        'Fold': fold + 1
    })
    feature_importances.append(fold_importance)

    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

# Calculate overall metrics
overall_auc = roc_auc_score(y, oof_predictions)
print(f"Average Validation AUC: {np.mean(valid_aucs):.5f}")
print(f"Overall OOF AUC: {overall_auc:.5f}")

# Save OOF predictions and true values
oof_df = X.copy()
oof_df['diagnosed_diabetes'] = y
oof_df['pred'] = oof_predictions
oof_df.to_csv(PATH+'Xgb/'+f"Xgboost_oof_predictions_auc_{overall_auc:.5f}.csv", index=False)

# Aggregate and save feature importances
feature_importances_df = pd.concat(feature_importances)
average_importance = feature_importances_df.groupby('Feature')['Importance'].mean().reset_index()
average_importance = average_importance.sort_values('Importance', ascending=False)
average_importance.to_csv(PATH+'Xgb/'+f"Xgboost_average_feature_importance.csv", index=False)

# Plot average feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=average_importance.head(25))
plt.title('Top Features (Average Importance)')
plt.tight_layout()
plt.show()

# Generate submission with averaged test predictions
test['diagnosed_diabetes'] = test_predictions
submission = test[['id', 'diagnosed_diabetes']]
submission.to_csv(PATH+'Xgb/'+f"Xgboost_submission_cv_auc_{overall_auc:.5f}.csv", index=False)