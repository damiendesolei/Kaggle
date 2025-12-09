# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:59:31 2025

@author: zrj-desktop
"""

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


#employment_status
def encode_employment_status(df):
    df['employment_status_Retired'] = df['employment_status'].apply(lambda x: 1 if x == 'Retired' else 0)
    df['employment_status_Unemployed'] = df['employment_status'].apply(lambda x: 1 if x == 'Unemployed' else 0)
    df['employment_status_Student'] = df['employment_status'].apply(lambda x: 1 if x == 'Student' else 0)
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
df = encode_employment_status(df)



#### FEATRURES ####
FEATURES = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'gender_Other',
       'gender_Female', 'ethnicity_Hispanic', 'ethnicity_Black',
       'ethnicity_Asian', 'ethnicity_Other', 'education_level_Graduate',
       'education_level_Postgraduate', 'education_level_No_formal',
       'income_level_Lower_Middle', 'income_level_Upper_Middle',
       'income_level_Low', 'income_level_High', 'smoking_status_Current',
       'smoking_status_Former',
       'employment_status_Retired',
       'employment_status_Unemployed', 'employment_status_Student']



#### Create local valid ####
train = df[df.label=='train']
test = df[df.label=='test']

tr, val = train_test_split(train, test_size=0.15, stratify=train['diagnosed_diabetes'], random_state=2025)
tr = tr.reset_index(drop=True)
val = val.reset_index(drop=True)
print('tr shape:', tr.shape)
print('val shape:', val.shape)



#### Hyper Param ####
STUDY = False
N_HOUR = 5

if STUDY:
    # Hyper parameter tuning
    def objective(trial):
        # Define hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 100, 500, step=100)
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'exact',
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
        }
    
        # Set up K-Fold cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2025)
        scores = []
    
        for train_idx, val_idx in skf.split(tr, tr['diagnosed_diabetes']):
            # Split data into training and validation sets
            X_train_fold, X_val_fold = tr[FEATURES].iloc[train_idx], tr[FEATURES].iloc[val_idx]
            y_train_fold, y_val_fold = tr['diagnosed_diabetes'].iloc[train_idx], tr['diagnosed_diabetes'].iloc[val_idx]
            
            # Create XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dvalid = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
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
        'n_estimators': 300, 
        'max_depth': 11, 
        'learning_rate': 0.019572698015238758, 
        'colsample_bytree': 0.8, 
        'subsample': 0.9, 
        'alpha': 0.019746604283135488, 
        'lambda': 0.03763923955255857,
        #'grow_policy': 'lossguide',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 2025,
        'verbosity': 0,
        'use_label_encoder': False,
        'early_stopping_rounds': 100
    }

# Load the training data
X = train[FEATURES]
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
        'n_estimators': best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        #'max_leaves': best_params['max_leaves'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample'],
        'reg_alpha': best_params['alpha'],
        'reg_lambda': best_params['lambda'],
        'tree_method': 'exact',
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

    # Initialize and train the model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        verbose=False
    )
    
    # Generate validation predictions
    y_pred_valid = model.predict_proba(X_valid_fold)[:, 1]  # Get probability of positive class
    fold_auc = roc_auc_score(y_valid_fold, y_pred_valid)
    valid_aucs.append(fold_auc)
    oof_predictions[valid_idx] = y_pred_valid

    # Generate test predictions and accumulate
    test_pred = model.predict_proba(test[FEATURES])[:, 1]
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