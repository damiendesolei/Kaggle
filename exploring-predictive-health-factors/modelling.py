# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:50:49 2025

@author: zrj-desktop
"""

import os
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np


from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score


import lightgbm as lgb
import optuna


import joblib 
import matplotlib.pyplot as plt


# If the local directory exists, use it; otherwise, use the Kaggle input directory
PATH = '/kaggle/input/exploring-predictive-health-factors' if os.path.exists('/kaggle/input/exploring-predictive-health-factors') else r'G:\\kaggle\exploring-predictive-health-factors\\'


# Read in data
train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')


# Fill in nan for Weight
train['Weight_kg'] = train['Weight_kg'].fillna(
    train.groupby(['Exercise_Frequency','Exercise_Type'])['Weight_kg'].transform('mean')
)

test['Weight_kg'] = test['Weight_kg'].fillna(
    test.groupby(['Exercise_Frequency','Exercise_Type'])['Weight_kg'].transform('mean')
)


# Map categorical variable to numeric
integer_map = {np.nan: 0,
               'No': 0,
               'Yes': 1,
}


hirsutism_dict = {np.nan: 0,
                  'No': 0,
                  'No, Yes, not diagnosed by a doctor': 0,
                  'Yes': 1,
                  'Yes, diagnosed by a doctor': 1
}


hormonal_dict = {np.nan: 0,           
                 'No': 0,
                 'No, Yes, not diagnosed by a doctor': 0,
                 'Yes': 1,
                 'Yes Significantly': 2
}


conception_dict = {np.nan: 0,
                   'No': 0,
                   'No, Yes, not diagnosed by a doctor': 0,
                   'Yes': 1,
                   'Yes, diagnosed by a doctor': 1,
                   'Somewhat': 0
}


insulin_dict = {np.nan: 0,
                'No': 0,
                'No, Yes, not diagnosed by a doctor': 0,
                'Yes': 1,
                'Yes Significantly': 2
}


exercise_frequency = {np.nan: 0,
                      'Never': 0,
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
                      '6-8 hours': 0
}


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


sleep_hours = {np.nan: 0,
               'Less than 6 hours': 0,
               '3-4 hours': 1,               
               '6-8 hours': 1,
               '9-12 hours': 2,
               'More than 12 hours': 3,
               
               '6-8 Times a Week': 1, #suspect typo
               '6-12 hours': 1,
               '20 minutes': 0,
}


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


age_dict = {'15-20': 0,
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
              
            np.nan: 0
}


def initial_feature_map(df):
    
    df['Age']=df['Age'].replace(age_dict)
    
    df['Hyperandrogenism']=df['Hyperandrogenism'].replace(integer_map)
    df['Hirsutism']=df['Hirsutism'].replace(hirsutism_dict)
    df['Hormonal_Imbalance']=df['Hormonal_Imbalance'].replace(hormonal_dict)
    df['Conception_Difficulty']=df['Conception_Difficulty'].replace(conception_dict)
    df['Insulin_Resistance']=df['Insulin_Resistance'].replace(insulin_dict)
    
    df['Exercise_Frequency']=df['Exercise_Frequency'].replace(exercise_frequency)
    df['Exercise_Duration']=df['Exercise_Duration'].replace(exercise_duration)
    df['Exercise_Type']=df['Exercise_Type'].replace(exercise_type)
    
    df['Sleep_Hours']=df['Sleep_Hours'].replace(sleep_hours)
    
    df['Exercise_Benefit']=df['Exercise_Benefit'].replace(exercise_benefit)

    return df


train['PCOS']=train['PCOS'].replace(integer_map)
train = initial_feature_map(train)
test = initial_feature_map(test)





# intial features with only numeric columns
features_0 = [col for col in test.columns if (test[col].dtype != 'object' and test[col].dtype != 'datetime64[ns]')]

# Drop ID which should not be used
remove_features = ['ID'] #+ remove_features

# Fetures for modelling
features = [feature for feature in features_0 if feature not in remove_features]




# Setup model name to tune and predict
model_name = f'lgb_{len(features)}_parameters'
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

# Hyper parameter tuning
def objective(trial):
    # Define hyperparameters
    param = {
        'objective': 'regression',  
        'metric': 'auc',  
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
        'max_depth': trial.suggest_int('max_depth', 1, 12, step=1),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  
        'num_leaves': trial.suggest_int('num_leaves', 2, 128, step=1), 
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0, step=0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0, step=0.1),
        #'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.001, 0.1),
        "device_type": "cpu",  
        "seed" : 2025
    }


    # Set up Stratified K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=2025)
    scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create LightGBM Dataset
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)
        
        # Train LightGBM model
        model = lgb.train(
            params=param,
            train_set=train_set,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(10)
            ]
        )

        # Predict on validation set
        y_pred = model.predict(X_val_fold)

        # Calculate RMSE (or another regression metric)
        auc = roc_auc_score(y_val_fold, y_pred)
        scores.append(auc)    


    mean_auc = np.mean(scores)

    return mean_auc


# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=3600*0.3, n_jobs=1) # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best AUC:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
file_name = model_path + model_name + f"_mae_{best_score:.4f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")



# best_params = {'n_estimators': 400,
#       'max_depth': 2,
#       'learning_rate': 0.0736307161560456,
#       'num_leaves': 60,
#       'feature_fraction': 0.7,
#       'bagging_fraction': 0.8,
#       'lambda_l1': 0.06435445123914581,
#       'lambda_l2': 0.039695211960611654}
# Best AUC: 0.9305892903392904


# Model fitting and prediction
model =lgb.LGBMRegressor(device='cpu', gpu_use_dp=True, objective='binary', **best_params) # from Hyper param tuning


# Train LightGBM model with early stopping and evaluation logging
model.fit(X_train, y_train,  
          eval_metric='auc'
          #eval_set=[(X_valid, y_valid)], 
          #callbacks=[
          #    lgb.early_stopping(100), 
          #    lgb.log_evaluation(10)
          #]
          )


# Append the trained model to the list
#models.append(model)

# Test on the local validation set
y_pred = model.predict(X_valid)
valid_auc = roc_auc_score(y_valid, y_pred)
print(f"valid auc: {valid_auc}")
#valid auc: 0.7875
  


# Check the prediction error
CHECK = True
if CHECK:
    tr = pd.read_csv(PATH+'train.csv')
    X_tr, X_val, y_tr, y_val = train_test_split(tr[features], tr['PCOS'], test_size=0.10, stratify=y, random_state=2025)
    X_val['PCOS'] = y_val
    X_val['pred'] = y_pred
    X_val.to_csv(model_path + f'{model_name}_error_{valid_auc:.5f}.csv', index=False)

    
# Save the trained model to a file
joblib.dump(model, model_path + f'{model_name}_auc_{valid_auc:.5f}.model')




# assess the feature importance
lgb.plot_importance(model, max_num_features=25)  # Limit to top 30 features
plt.show()
    

# Create a DataFrame
lgb_feature_importance= pd.DataFrame({
    'Feature': model.feature_name_,
    'Importance': model.feature_importances_
})

lgb_feature_importance = lgb_feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
lgb_feature_importance.to_csv(model_path + f'{model_name}_features_{valid_auc:.4f}.csv', index=False)






# Predict and submit
test['PCOS'] = model.predict(test[features])


submission = test[['ID','PCOS']]
submission.to_csv(model_path + f"{model_name}_submission_{valid_auc:.4f}.csv",index=False)


