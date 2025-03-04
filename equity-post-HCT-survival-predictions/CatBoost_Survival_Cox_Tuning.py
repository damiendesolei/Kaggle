# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:59:35 2025

@author: zrj-desktop
"""

import lifelines
import time

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 500)



# Load the data
PATH = r'G:\\kaggle\equity-post-HCT-survival-predictions\\'

test = pd.read_csv(PATH+"test.csv")
print("Test shape:", test.shape )

train = pd.read_csv(PATH+"train.csv")
print("Train shape:",train.shape)
train.head()



# # Transfor efs and efs_time
# from lifelines import KaplanMeierFitter
# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(df[time_col], df[event_col])
#     y = kmf.survival_function_at_times(df[time_col]).values
#     return y
# train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')



# Combine train and test
combined = pd.concat([train, test], axis=0, ignore_index=True)
print("Combined data shape:", combined.shape )



# Features 1
print("< deal with outlier >")
combined['nan_value_each_row'] = combined.isnull().sum(axis=1)
#year_hct=2020 only 4 rows.
combined['year_hct']=combined['year_hct'].replace(2020,2019)
combined['age_group']=combined['age_at_hct']//10
combined['donor_age_group']=combined['donor_age']//10

#karnofsky_score 40 only 10 rows.
combined['karnofsky_score']=combined['karnofsky_score'].replace(40,50)
#hla_high_res_8=2 only 2 rows.
combined['hla_high_res_8']=combined['hla_high_res_8'].replace(2,3)
#hla_high_res_6=0 only 1 row.
combined['hla_high_res_6']=combined['hla_high_res_6'].replace(0,2)
#hla_high_res_10=3 only 1 row.
combined['hla_high_res_10']=combined['hla_high_res_10'].replace(3,4)
#hla_low_res_8=2 only 1 row.
combined['hla_low_res_8']=combined['hla_low_res_8'].replace(2,3)
# N/A - disease not classifiable; N/A - non-malignant indication;  N/A - pediatric
combined['dri_score']=combined['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
combined['dri_score_NA']=combined['dri_score'].apply(lambda x:int('N/A' in str(x)))

for col in ['diabetes','pulm_moderate','cardiac']:
    combined.loc[combined[col].isna(),col]='Not done'

print("< cross feature >")
combined['donor_age+age_at_hct']=combined['donor_age']+combined['age_at_hct']
combined['donor_age-age_at_hct']=combined['donor_age']-combined['age_at_hct']
combined['donor_age*age_at_hct']=combined['donor_age']*combined['age_at_hct']
combined['donor_age/age_at_hct']=combined['donor_age']/combined['age_at_hct']

# combined['comorbidity_score+karnofsky_score']=combined['comorbidity_score']+combined['karnofsky_score']
combined['comorbidity_score-karnofsky_score']=combined['comorbidity_score']-combined['karnofsky_score']
# combined['comorbidity_score*karnofsky_score']=combined['comorbidity_score']*combined['karnofsky_score']
# combined['comorbidity_score/karnofsky_score']=combined['comorbidity_score']/combined['karnofsky_score']

# combined['age*(comorbidity_score+karnofsky_score)']=combined['age_at_hct']*(combined['comorbidity_score']+combined['karnofsky_score'])
# combined['age*(comorbidity_score-karnofsky_score)']=combined['age_at_hct']*(combined['comorbidity_score']-combined['karnofsky_score'])
# combined['age*(comorbidity_score*karnofsky_score)']=combined['age_at_hct']*(combined['comorbidity_score']*combined['karnofsky_score'])
combined['age*(comorbidity_score/karnofsky_score)']=combined['age_at_hct']*(combined['comorbidity_score']/combined['karnofsky_score'])



# Features 2
# seasonal features
combined['cos_year'] = np.cos(combined['year_hct'] * (2 * np.pi) / 100)
combined['sin_year'] = np.sin(combined['year_hct'] * (2 * np.pi) / 100)



# Features 3
# find categorical features for interaction
#nunique=2
nunique2=[col for col in combined.columns if combined[col].nunique()==2 and col!='efs']
#nunique<50
nunique50=[col for col in combined.columns if combined[col].nunique()<50 and col not in ['efs','weight']]+['age_group','dri_score_NA']


print("< combine category feature >")
for i in range(len(nunique2)):
    for j in range(i+1,len(nunique2)):
        combined[nunique2[i]+'_'+nunique2[j]]=combined[nunique2[i]].astype(str)+'_'+combined[nunique2[j]].astype(str)

combine_category_cols=[]
for i in range(len(nunique2)):
    for j in range(i+1,len(nunique2)):
        combine_category_cols.append(nunique2[i]+'_'+nunique2[j])


print('Category features with only 2 values:', nunique2) # 4 features -> C(9, 2) = 36 new CAT features
print(f'{len(combine_category_cols)} newly created combined features:', combine_category_cols)



# Convert all category columns to object
combined = combined.astype({col: 'object' for col in combined.select_dtypes('category').columns})
combined.info()

# remove non features columns
RMV = ["ID","efs","efs_time","y"] #remove id and targets
FEATURES = [c for c in combined.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

# CAT features
CATS = []
for c in FEATURES:
    if combined[c].dtype=="object":  # replace nan with NAN string for all categorical columns
        CATS.append(c)
        #train[c] = train[c].fillna("NAN")
        #test[c] = test[c].fillna("NAN")
        combined[c] = combined[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")



# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min() # ensure feature starts with 0
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
combined.info()



# Fill nan for Age
combined[['donor_age', 'donor_age_group']] = combined[['donor_age', 'donor_age_group']].fillna(0)



# split into train and test
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

train.tail()



# Target encoding functions
#reference: https://www.kaggle.com/code/cdeotte/first-place-single-model-cv-1-016-lb-1-016  
def target_encode(train, valid, test, col, target="y", kfold=10, smooth=20, agg="mean"):

    train['kfold'] = ((train.index) % kfold)
    col_name = '_'.join(col)
    train[f'TE_{agg.upper()}_' + col_name] = 0. # Initialize encoded column
    
    for i in range(kfold):
        
        df_tmp = train[train['kfold']!=i]
        # Compute GLOBAL stats for smoothing
        if agg=="mean": mn = train[target].mean()
        elif agg=="median": mn = train[target].median()
        elif agg=="min": mn = train[target].min()
        elif agg=="max": mn = train[target].max()
        elif agg=="nunique": mn = 0
        elif agg=="skew": mn = train[target].skew()  # global skewness
        elif agg == "std": mn = train[target].std()   # global std
        # Compute the count for smoothing
        df_tmp = df_tmp[col + [target]].groupby(col, observed=False).agg([agg, 'count']).reset_index()
        df_tmp.columns = col + [agg, 'count']

        # Handle nunique separately
        if agg=="nunique":
            df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
        else:
            df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth) #smoothing
        # Apply the encoding to the training set    
        df_tmp_m = train[col + ['kfold', f'TE_{agg.upper()}_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        df_tmp_m.loc[df_tmp_m['kfold']==i, f'TE_{agg.upper()}_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_tmp']
        train[f'TE_{agg.upper()}_' + col_name] = df_tmp_m[f'TE_{agg.upper()}_' + col_name].fillna(mn).values  
        
    # Apply the same transformation to the full training set, validation, and test
    df_tmp = train[col + [target]].groupby(col,observed=False).agg([agg, 'count']).reset_index()
    if agg=="mean": mn = train[target].mean()
    elif agg=="median": mn = train[target].median()
    elif agg=="min": mn = train[target].min()
    elif agg=="max": mn = train[target].max()
    elif agg=="nunique": mn = 0
    elif agg == "skew": mn = train[target].skew()  # skewness
    elif agg == "std": mn = train[target].std()  # std
    df_tmp.columns = col + [agg, 'count']
    
    if agg=="nunique":
        df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
    else:
        df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)

    # apply encoding to valid and test sets
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_{agg.upper()}_' + col_name] = valid[f'TE_{agg.upper()}_' + col_name].astype("float32")

    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_{agg.upper()}_' + col_name] = test[f'TE_{agg.upper()}_' + col_name].astype("float32")

    train = train.drop('kfold', axis=1)
    train[f'TE_{agg.upper()}_' + col_name] = train[f'TE_{agg.upper()}_' + col_name].astype("float32")

    return(train, valid, test)



# List of interaction features for target encoding
lists2 = [['prim_disease_hct', 'donor_age_group'], 
          ['prim_disease_hct', 'age_group'],
          ['dri_score','donor_age_group'], 
          ['dri_score','age_group'],
          ['conditioning_intensity', 'donor_age_group'],
          ['conditioning_intensity', 'age_group'],
          ['comorbidity_score', 'age_group'],
          ['prim_disease_hct'],
          ['dri_score'],
          ['conditioning_intensity'],
          ['cyto_score'],
          ['year_hct']
]
print(f"We have {len(lists2)} powerful combination of columns for Target Encoding!")





#### CatBoost with Survival:Cox ####

# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1

# Hyper parameter tuning
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

# from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


# Define the parameter space
def objective(trial):
    
    param = {
        'loss_function': 'RMSE', 
        'grow_policy': 'Depthwise',#trial.suggest_categorical('grow_policy', ['Depthwise', 'Lossguide']),
        'task_type': 'GPU',  
        #'gpu_use_dp': True,

        'n_estimators': 5000,
        #'iterations': trial.suggest_int('iterations', 3000, 5000, step=200),
        'depth': trial.suggest_int('depth', 8, 16, step=1),   # max depth is 16
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 128, 512, step=2),
        
        #'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.6, 1.0), # Random Subspace Method (rsm not supported on GPU)
        #'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.001, 0.1, log=True),

        'random_seed': 2025,
        'logging_level': "Silent"  # Suppress CatBoost logs
    }

    # Time series cross-validation
    skf = StratifiedKFold(n_splits=4, shuffle=False)
    scores = []
    
    for i, (train_index, test_index) in enumerate(skf.split(train, train["efs"])):       
        #x_train = train.loc[train_index,FEATURES].copy()
        x_train = train.loc[train_index,FEATURES+["efs_time2"] ].copy()
        y_train = train.loc[train_index,"efs_time2"]
        x_valid = train.loc[test_index,FEATURES].copy()
        y_valid = train.loc[test_index,"efs_time2"]
        x_test = test[FEATURES].copy()
        
        #https://www.kaggle.com/code/cdeotte/first-place-single-model-cv-1-016-lb-1-016
        start = time.time()
        #print(f"FEATURE ENGINEER {len(FEATURES)} COLUMNS and {len(lists2)} GROUPS: ",end="")
        for j,f in enumerate(FEATURES+lists2):
    
            if j<len(FEATURES): c = [f]
            else: c = f 
    
            # HIGH CARDINALITY FEATURES - TE MIN, MAX, NUNIQUE and CE
            if j>=len(FEATURES):
                x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, target="efs_time2", smooth=20, agg="mean")
                x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, target="efs_time2", smooth=0, agg="std")
                x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, target="efs_time2", smooth=0, agg="skew")
        
        end = time.time()
        elapsed = end-start
        #print(f"Feature engineering took {elapsed:.1f} seconds")
        x_train = x_train.drop("efs_time2",axis=1)
        #print(x_train.info())
        #print(x_train.tail())


        # Create CatBoost pools
        dtrain = cb.Pool(x_train, label=y_train, cat_features=CATS)
        dvalid = cb.Pool(x_valid, label=y_valid, cat_features=CATS)

        # Train Catboost model
        model = cb.CatBoostRegressor(**param)
        model = model.fit(
            dtrain,
            eval_set=dvalid,
            early_stopping_rounds=100,
            use_best_model=True
        )
        # Predict on validation set
        y_pred = model.predict(dvalid)
    
        mae = mean_absolute_error(y_valid, y_pred)  # WMAE for regression
        scores.append(mae)  
    
    mean_mae = np.mean(scores)
    
    return mean_mae


# Run Optuna study
N_HOUR = 6
CORES = 1

print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=3600*N_HOUR, n_jobs=CORES)  # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best mae:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
OUT_PATH = r'G:\\kaggle\equity-post-HCT-survival-predictions\models\\'
file_name = f"Cb_with_Survival_Cox_mae_{best_score:.6f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(OUT_PATH+file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")