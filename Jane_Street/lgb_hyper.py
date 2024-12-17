# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:51:41 2024

@author: zrj-desktop
"""

import os
import joblib 
import itertools

import pandas as pd
import polars as pl
import numpy as np 
import scipy


import lightgbm as lgb
import matplotlib.pyplot as plt


from joblib import Parallel, delayed




def reduce_mem_usage(self, float16_as32=True):
    #memory_usage()是df每列的内存使用量,sum是对它们求和, B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:#遍历每列的列名
        col_type = df[col].dtype#列名的type
        if col_type != object and str(col_type)!='category':#不是object也就是说这里处理的是数值类型的变量
            c_min,c_max = df[col].min(),df[col].max() #求出这列的最大值和最小值
            if str(col_type)[:3] == 'int':#如果是int类型的变量,不管是int8,int16,int32还是int64
                #如果这列的取值范围是在int8的取值范围内,那就对类型进行转换 (-128 到 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #如果这列的取值范围是在int16的取值范围内,那就对类型进行转换(-32,768 到 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                #如果这列的取值范围是在int32的取值范围内,那就对类型进行转换(-2,147,483,648到2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                #如果这列的取值范围是在int64的取值范围内,那就对类型进行转换(-9,223,372,036,854,775,808到9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:#如果是浮点数类型.
                #如果数值在float16的取值范围内,如果觉得需要更高精度可以考虑float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:#如果数据需要更高的精度可以选择float32
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)  
                #如果数值在float32的取值范围内，对它进行类型转换
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                #如果数值在float64的取值范围内，对它进行类型转换
                else:
                    df[col] = df[col].astype(np.float64)
    #计算一下结束后的内存
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #相比一开始的内存减少了百分之多少
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



# Define the path to the input data directory
# If the local directory exists, use it; otherwise, use the Kaggle input directory
input_path = './jane-street-real-time-market-data-forecasting/' if os.path.exists('./jane-street-real-time-market-data-forecasting') else r'G:\\kaggle\jane-street-real-time-market-data-forecasting\\'


# Define the feature names based on the number of features (79 in this case)
feature_names = [f"feature_{i:02d}" for i in range(79)]

# Number of validation dates to use
num_valid_dates = 100

# Number of dates to skip from the beginning of the dataset
skip_dates = 1000 #skil roughly 3 years, keeping most recent 800 days


# Load the training data
df = pd.read_parquet(f'{input_path}train.parquet')
    
df = reduce_mem_usage(df, False)
    
df = df[df['date_id'] >= skip_dates].reset_index(drop=True)

dates = df['date_id'].unique()
valid_dates = dates[-num_valid_dates:]
train_dates = dates[:-num_valid_dates]

print(df.tail())

    
    

# Create a directory to store the trained models
os.system('mkdir models')
os.getcwd()

# Define the path to load pre-trained models (if not in training mode)
model_path = '/kaggle/input/jsbaselinezyz' if os.path.exists('/kaggle/input/jsbaselinezyz') else r'C:\\Users\\zrj-desktop\\models\\'
#os.path.exists(r'G:\\kaggle\jane-street-real-time-market-data-forecasting\input\\')

#set up random column
np.random.seed(24)
df['random'] = np.random.rand(df.shape[0])
feature_names.append('random')


# new features of column difference
feature_names_0 = [f"{i:02d}" for i in range(79)]
feature_combination = itertools.combinations(feature_names_0, 2)

def columns_diff(df, combinations):
    for i, j in combinations:
        df[f'diff_{i}_{j}'] = df[f'feature_{i}'] - df[f'feature_{j}']
    return df


# If in training mode, prepare validation data
# Extract features, target, and weights for validation dates
X_valid = df[feature_names].loc[df['date_id'].isin(valid_dates)]
y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)]
w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)]




    
    
# additional descriptive features
def descriptive_stat(df, feature):
    #features = df.columns.tolist()
    features = feature
    df['mean_features'] = df[features].mean(axis=1)
    df['std_features'] = df[features].std(axis=1)
    df['max_features'] = df[features].max(axis=1)
    df['min_features'] = df[features].min(axis=1)
    df['median_features'] = df[features].median(axis=1)
    df['range_features'] = df['max_features'] - df['min_features']
    df['90percentile_features'] = np.percentile(df[features], 90, axis=1)
    df['75percentile_features'] = np.percentile(df[features], 75, axis=1)
    df['25percentile_features'] = np.percentile(df[features], 25, axis=1)
    df['10percentile_features'] = np.percentile(df[features], 10, axis=1)
    df['kurtosis_features'] = scipy.stats.kurtosis(df[features], axis=1)
    df['skew_features'] = scipy.stats.skew(df[features], axis=1)
    
    mean_abs_dev = (df[features] - df[features].mean(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
    median_abs_dev = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
    range_abs_diff = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().max(axis=1) - (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().min(axis=1)
    geometric_mean = np.exp(np.log(df[features].replace(0, 1)).mean(axis=1))
    harmonic_mean = len(features) / (1 / df[features].replace(0, 1)).sum(axis=1)
    coeff_variation = df['std_features'] / df['mean_features']
    df['mean_absolute_deviation'] = mean_abs_dev
    df['median_absolute_deviation'] = median_abs_dev
    df['range_abs_diff'] = range_abs_diff
    df['geometric_mean'] = geometric_mean
    df['harmonic_mean'] = harmonic_mean
    df['coeff_variation'] = coeff_variation
    # just keep the descriptive statistics
    #dataset = df.drop(features, axis=1)
    
    return df



# Initialize a list to store trained models
models = []


# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return 'r2', r2, True


# Number of folds for cross-validation
N_fold = 1
#i = 0

# Function to train a model or load a pre-trained model
model_name = 'lgb_initial_with_random'
# Select dates for training based on the fold number
i=0

selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_fold == i]

# Specify model
model =lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2')

# Extract features, target, and weights for the selected training dates
X_train = df[feature_names].loc[df['date_id'].isin(selected_dates)]
y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)]
w_train = df['weight'].loc[df['date_id'].isin(selected_dates)]

# Train the model based on the type (LightGBM, XGBoost, or CatBoost)
# Train LightGBM model with early stopping and evaluation logging
model.fit(X_train, y_train, w_train,  
          eval_metric=[r2_lgb],
          eval_set=[(X_valid, y_valid, w_valid)], 
          callbacks=[
              lgb.early_stopping(100), 
              lgb.log_evaluation(10)
          ])


# Append the trained model to the list
#models.append(model)

# Save the trained model to a file
joblib.dump(model, f'./models/{model_name}_{i}.model')


# Collect garbage to free up memory
import gc
gc.collect()



#assess the feature importance
lgb.plot_importance(model, max_num_features=50)  # Limit to top 10 features
plt.show()
    


# Create a DataFrame
lgb_feature_importance= pd.DataFrame({
    'Feature': model.feature_name_,
    'Importance': model.feature_importances_
})

lgb_feature_importance = lgb_feature_importance.sort_values('Importance')



