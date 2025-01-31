# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 20:50:49 2025

@author: zrj-desktop
"""

import os
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np


from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score


import xgboost as xgb
import optuna


import joblib 
import matplotlib.pyplot as plt
import seaborn as sns


# If the local directory exists, use it; otherwise, use the Kaggle input directory
PATH = '/kaggle/input/exploring-predictive-health-factors' if os.path.exists('/kaggle/input/exploring-predictive-health-factors') else r'G:\\kaggle\exploring-predictive-health-factors\\'


# Read in data
train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')



# Fill in nan for Weight
train['Weight_kg'] = train['Weight_kg'].fillna(
    train.groupby(['Hyperandrogenism','Exercise_Frequency','Exercise_Type'])['Weight_kg'].transform('mean')
)

test['Weight_kg'] = test['Weight_kg'].fillna(
    test.groupby(['Hyperandrogenism','Exercise_Frequency','Exercise_Type'])['Weight_kg'].transform('mean')
)

# Check nan
print(f'Train has {train.Weight_kg.isna().sum()} nan in Weight_kg')
print(f'Test has {test.Weight_kg.isna().sum()} nan in Weight_kg')


# Fill in na for Hyperandrogenism
# Fill nan for Hyperandrogenism
train['Hyperandrogenism'] = train['Hyperandrogenism'].fillna(
    train.groupby(['Age','Hormonal_Imbalance','Exercise_Duration'])['Hyperandrogenism'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
)

test['Hyperandrogenism'] = test['Hyperandrogenism'].fillna(
    test.groupby(['Age','Hormonal_Imbalance','Exercise_Duration'])['Hyperandrogenism'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
)

# Check nan
print(f'Train has {train.Hyperandrogenism.isna().sum()} nan in Hyperandrogenism')
print(f'Test has {test.Hyperandrogenism.isna().sum()} nan in Hyperandrogenism')



# Map categorical variable to numeric
integer_map = {'Yes': 1,
               'Yes, diagnosed by a doctor': 1,
               'Yes Significantly': 2,
               'No': 0,
               'No, Yes, not diagnosed by a doctor': 0,
               'Somewhat': 0,
               np.nan: 0}


exercise_frequency = {'Never': 0,
                      'Rarely': 1,
                      
                      '1-2 Times a Week': 2,
                      'Less than 6-8 Times a Week': 2,
                      'Less than usual': 2,
                      'Less than 6 hours': 2,
                      
                      '3-4 Times a Week': 3,
                      '6-8 Times a Week': 4,

                      'Daily': 4,
                      'Somewhat': 1,
                      '1/2 Times a Week': 2,
                      
                      '30-35': 0,
                      '6-8 hours': 0,
                      np.nan: 0}


exercise_duration = {'Not Applicable': 0,
                     'Less than 30 minutes': 1,
                     '20 minutes': 1,
                     '30 minutes': 2,
                     '45 minutes': 3,
                     'More than 30 minutes': 4,
                     '30 minutes to 1 hour': 5,
                     'Less than 6 hours': 6,
                     
                     '6-8 hours': 7,
                     'Less than 20 minutes': 1,
                     'Not Much': 1,
                     
                     '40 minutes': 3,
                     '3-4 Times a Week': 0,
                     '1-2 Times a Week': 0,
                     np.nan: 0}


sleep_hours = {'Less than 6 hours': 0,
               '3-4 hours': 1,               
               '6-8 hours': 1,
               '9-12 hours': 2,
               'More than 12 hours': 3,
               
               '6-8 Times a Week': 1, #suspect typo
               '6-12 hours': 1,
               '20 minutes': 0,
               np.nan: 0}


exercise_benefit = {'Not at All': 0,
                    'Not Much': 1,
                    'Somewhat': 2,
                    'Yes Significantly': 3,
                    
                    np.nan: 0}

exercise_type = {'No Exercise': 0,
                 'Somewhat': 0,                 
 
                 'Flexibility and balance (e.g.': 1,
                 'Flexibility and balance (e.g., yoga, pilates), None': 1,
                 'Flexibility and balance (e.g., yoga, pilates)': 1,

                 'Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)': 2,
                 'Strength training (e.g., weightlifting, resistance exercises)': 2,
                 'Strength training': 2, 
                 'Strength training (e.g.': 2, 
                 
                 'Cardio (e.g.': 3,
                 'Cardio (e.g., running, cycling, swimming)': 3, 
                 'Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)': 3,
                 'Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)': 3,
                 'Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)': 3,
                 'Cardio (e.g., running, cycling, swimming), None': 3, 
                 
                 'High-intensity interval training (HIIT)': 4,

                 'Strength (e.g.': 2,
                 'Yes Significantly': 4,
                 'No': 0,
                 'Sleep_Benefit': 0,
                 'Not Applicable': 0,
                 np.nan: 0}


age = {'15-20': 0,
       'Less than 20)': 0,
       'Less than 20': 0,
       'Less than 20-25': 0,
       '20': 0,
       '22-25': 1,
       '20-25': 1,
       '25-25': 1,
       '25-30': 2,
       '30-25': 2,
       '30-40': 3,
       '30-35': 3,
       '30-30': 3,
       '35-44': 4,
       '45 and above': 5,
       '45-49': 5,
       '50-60': 6,
          
       np.nan: 0}


def initial_feature_map(df):
    
    df['Age']=df['Age'].apply(lambda x:age[x])
    
    df['Hyperandrogenism']=df['Hyperandrogenism'].apply(lambda x:integer_map[x])
    df['Hirsutism']=df['Hirsutism'].apply(lambda x:integer_map[x])
    df['Hormonal_Imbalance']=df['Hormonal_Imbalance'].apply(lambda x:integer_map[x])
    df['Conception_Difficulty']=df['Conception_Difficulty'].apply(lambda x:integer_map[x])
    df['Insulin_Resistance']=df['Insulin_Resistance'].apply(lambda x:integer_map[x])
    
    df['Exercise_Frequency']=df['Exercise_Frequency'].apply(lambda x:exercise_frequency[x])
    df['Exercise_Duration']=df['Exercise_Duration'].apply(lambda x:exercise_duration[x])
    df['Exercise_Type']=df['Exercise_Type'].apply(lambda x:exercise_type[x])
    
    df['Sleep_Hours']=df['Sleep_Hours'].apply(lambda x:sleep_hours[x])
    
    df['Exercise_Benefit']=df['Exercise_Benefit'].apply(lambda x:exercise_benefit[x])

    return df


train['PCOS']=train['PCOS'].apply(lambda x:integer_map[x])
train_0 = train

train = initial_feature_map(train)
test = initial_feature_map(test)


# one-hot encode Hyperandrogenism
#train['Hyperandrogenism_Yes'] = np.where(train['Hyperandrogenism'] == 1, 1, 0)



# intial features with only numeric columns
features_0 = [col for col in test.columns if (test[col].dtype != 'object' and test[col].dtype != 'datetime64[ns]')]

# Drop ID which should not be used
remove_features = ['ID'] #+ remove_features

# Fetures for modelling
features = [feature for feature in features_0 if feature not in remove_features]




# Setup model name to tune and predict
model_name = f'xgb_{len(features)}_parameters'
model_path = r'G:\\kaggle\exploring-predictive-health-factors\model\\'



# Split the data into train and valiation
X = train[features]
y = train['PCOS']

# Reserve 10% of data as local validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, stratify=y, random_state=2025)

# Reset index
X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)



STUDY = False
N_HOUR = 1

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
            
            'max_depth': trial.suggest_int('max_depth', 1, 12, step=1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            #'max_leaves': trial.suggest_int('max_leaves', 2, 128, step=1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'alpha': trial.suggest_loguniform("alpha", 0.001, 0.1),
            'lambda': trial.suggest_loguniform("lambda", 0.001, 0.1),
            
            'seed': 2025,
            'verbosity': 0,
            "disable_default_eval_metric": 1,  # Disable default eval metric logs
        }
    
        # Set up K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=2025)
        scores = []
    
        for train_idx, val_idx in kf.split(X_train, y_train):
            # Split data into training and validation sets
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dvalid = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            # Train XGBoost model
            model = xgb.train(
                params=param,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=[(dvalid, 'eval')],
                #early_stopping_rounds=100,
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
    study.optimize(objective, timeout=3600*N_HOUR, n_jobs=1)  # 3600*n hour
    
    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best AUC:", study.best_value)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Format the file name with the best score
    file_name = model_path + model_name + f"_auc_{best_score:.4f}.csv"
    
    # Save the best parameters to a CSV file
    df_param = pd.DataFrame([best_params])  # Convert to DataFrame
    df_param.to_csv(file_name, index=False)  # Save to CSV
    
    print(f"Best parameters saved to {file_name}")



if not STUDY:
    xgb_params = {
        'n_estimators': 500, 
        'max_depth': 2, 
        'learning_rate': 0.012376787407810667, 
        'colsample_bytree': 0.7, 
        'subsample': 1.0, 
        'alpha': 0.012847356155418035, 
        'lambda': 0.002437123748008193,
        #'grow_policy': 'lossguide',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 2025,
        'verbosity': 0,
        'use_label_encoder': False
    }

# Load the training data
X = X_train
y = y_train

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

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
        'use_label_encoder': False
    }

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}")
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    # Initialize and train the model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_fold, y_train_fold)

    # Generate validation predictions
    y_pred_valid = model.predict_proba(X_valid_fold)[:, 1]  # Get probability of positive class
    fold_auc = roc_auc_score(y_valid_fold, y_pred_valid)
    valid_aucs.append(fold_auc)
    oof_predictions[valid_idx] = y_pred_valid

    # Generate test predictions and accumulate
    test_pred = model.predict_proba(test[features])[:, 1]
    test_predictions += test_pred / skf.n_splits

    # Save model and feature importance
    joblib.dump(model, f"{model_path}{model_name}_fold{fold+1}_auc_{fold_auc:.5f}.model")
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
oof_df['PCOS'] = y
oof_df['pred'] = oof_predictions
oof_df.to_csv(f"{model_path}{model_name}_oof_predictions_auc_{overall_auc:.5f}.csv", index=False)

# Aggregate and save feature importances
feature_importances_df = pd.concat(feature_importances)
average_importance = feature_importances_df.groupby('Feature')['Importance'].mean().reset_index()
average_importance = average_importance.sort_values('Importance', ascending=False)
average_importance.to_csv(f"{model_path}{model_name}_average_feature_importance.csv", index=False)

# Plot average feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=average_importance.head(25))
plt.title('Top Features (Average Importance)')
plt.tight_layout()
plt.show()

# Generate submission with averaged test predictions
test['PCOS'] = test_predictions
submission = test[['ID', 'PCOS']]
submission.to_csv(f"{model_path}{model_name}_submission_cv_auc_{overall_auc:.4f}.csv", index=False)