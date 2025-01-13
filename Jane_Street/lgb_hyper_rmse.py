# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:51:41 2024

@author: zrj-desktop
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib 
import itertools

import pandas as pd
pd.set_option('display.max_columns', None)
import polars as pl
import numpy as np 
#import stats
import scipy

from sklearn.linear_model import LinearRegression


import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


from tqdm import tqdm, tqdm_notebook
from joblib import Parallel, delayed


import optuna


        
        

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

# Number of dates to skip from the beginning of the dataset total 1700 days
skip_dates = 1100 #keeping most recent 600 days


# Load the training data
# df = pd.read_parquet(f'{input_path}train.parquet')
# https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags/notebook
class CONFIG:
    target_col = "responder_6"
    lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
    lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
    #valid_ratio = 0.05
    start_dt = 1100

# Use last 2 parquets
train = pl.scan_parquet(f'{input_path}train.parquet'
).select(
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"), # a new "id" column containing row indices starting from 0
    pl.all(), # all original columns.
).filter(
    pl.col("date_id").gt(CONFIG.start_dt)
)
    
    
# create lag feature
lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date
lags


# join the lag data to train
train = train.join(lags, on=["date_id", "symbol_id"],  how="left"
                   ).filter(pl.col("date_id").gt(CONFIG.start_dt+1)) # remove null rows for lagged responders
train


# Train test split
# len_train   = train.select(pl.col("date_id")).collect().shape[0]
# #valid_records = int(len_train * CONFIG.valid_ratio)

# last_100_unique_dates = (
#     train.select("date_id")
#     .unique(keep="last")  # Get unique values, keeping the last occurrence
#     .sort("date_id", descending=False)  # Sort in ascending order
#     .tail(100)  # Keep only the last 100 unique values
#     .collect()
#     .to_series()  # Convert to a Series for filtering
# )
# #len_ofl_mdl = len_train - valid_records
# #last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

# #print(f"\n len_train = {len_train}")
# #print(f"\n len_validation = {len_ofl_mdl}")
# #print(f"\n---> Last offline train date = {last_tr_dt}\n")



# #training_data = train.filter(pl.col("date_id").le(last_tr_dt))
# #validation_data   = train.filter(pl.col("date_id").gt(last_tr_dt))
# training_data = train.filter(~pl.col("date_id").is_in(last_100_unique_dates))
# validation_data = train.filter(pl.col("date_id").is_in(last_100_unique_dates)) # pick last 100 days for validation
    


df = train.collect().to_pandas()
#df = reduce_mem_usage(df, False)
feature_lagged_responders =  [f"responder_{idx}_lag_1" for idx in range(9)]
feature_names = feature_names + feature_lagged_responders 

#df = df[df['date_id'] >= skip_dates].reset_index(drop=True)

dates = df['date_id'].unique()
valid_dates = dates[-num_valid_dates:]
train_dates = dates[:-num_valid_dates]

#print(df.tail())

  
#set up random column
def add_random_column(df):
    np.random.seed(24)
    df['random'] = np.random.rand(df.shape[0])
    return df

add_random_column(df)
feature_names.append('random')
  
    

# Create a directory to store the trained models
os.system('mkdir models')
os.getcwd()

# Define the path to load pre-trained models (if not in training mode)
model_path = '/kaggle/input/jsbaselinezyz' if os.path.exists('/kaggle/input/jsbaselinezyz') else r'C:\\Users\\zrj-desktop\\models\\'
#os.path.exists(r'G:\\kaggle\jane-street-real-time-market-data-forecasting\input\\')


# combination_1 = pd.read_csv(model_path + "lgb_random_with_diff_combination_1_0(l2_0.646132_r2_0.00339201).csv")
# combination_2 = pd.read_csv(model_path + "lgb_random_with_diff_combination_2_0(l2_0.646372_r2_0.00302101).csv")
# combination_3 = pd.read_csv(model_path + "lgb_random_with_diff_combination_3_0(l2_0.645615_r2_0.00418869).csv")

# comb_features = list(combination_1[(combination_1.Importance>0) & (combination_1.Feature.str.len()>10)]['Feature']) \
#     + list(combination_2[(combination_2.Importance>0) & (combination_2.Feature.str.len()>10)]['Feature']) \
#     + list(combination_3[(combination_3.Importance>0) & (combination_3.Feature.str.len()>10)]['Feature'])

comb = pd.read_csv(model_path + "lgb_with_diff_comb_plus_lag_plus_roll_99_hyper_0_0.007095.csv")
#find the features with importance >= random
comb_features = list(comb[(comb.Importance>=1) & (comb.Feature.str.len()==26)]['Feature']) 

combinations = [
    ('feature_'+element.split('_')[2], 'feature_'+element.split('_')[4]) 
    for element in comb_features
]


#overwrite to reduce the number of diff features as the LB is showing negative score
combinations = [('feature_07', 'feature_56'), ('feature_07', 'feature_60'), ('feature_04', 'feature_60')
                , ('feature_09', 'feature_28'), ('feature_06', 'feature_59'), ('feature_15', 'feature_36')]


# # new features of column difference
# feature_names_0 = [f"feature_{i:02d}" for i in range(79)] 
# feature_combination = list(itertools.combinations(feature_names_0, 2))

# # Calculate chunk sizes for splitting
# total_combinations = len(feature_combination)
# chunk_size = total_combinations // 3
# remainder = total_combinations % 3

# # Split the combinations into three parts
# start = 0
# feature_combination_1 = feature_combination[start : start + chunk_size + (1 if remainder > 0 else 0)]
# start += len(feature_combination_1)
# feature_combination_2 = feature_combination[start : start + chunk_size + (1 if remainder > 1 else 0)]
# start += len(feature_combination_2)
# feature_combination_3 = feature_combination[start:]



def columns_diff(df, combinations):
    for i, j in combinations:
        #print(f'feature_{i} - feature_{j} is done..')
        df[f'diff_{i}_{j}'] = df[i] - df[j]
    print('Ratio features are created...')
    return df

# create new columns   
columns_diff(df, combinations)
#df = reduce_mem_usage(df, False)

# find the new column names - one after random
new_diff_cols = list(df.columns[df.columns.get_loc('random')+1:])

feature_names = feature_names + new_diff_cols

# override feature names to top 
# feature_names = comb_features = list(comb[comb.Importance>=11]['Feature']) #top 117 features
# feature_names.append('random')


# new features source:
# https://www.kaggle.com/code/jsaguiar/baseline-with-multiple-models?scriptVersionId=15034696

def calculate_slope(series_y):
    n = len(series_y)
    if n < 2:  # Ensure at least two points are available for regression
        return 0
    x = np.arange(n)  # Use a simple range as x values since intervals are consistent
    slope, _ = np.polyfit(x, series_y, 1)  # Linear regression
    
    return slope


 
# https://www.kaggle.com/code/wimwim/rolling-quantiles
# def create_rolling_features(df, feature, rows=250000):
#     segments = int(np.floor(df.shape[0] / rows))
    
#     for segment in tqdm_notebook(range(segments)):
#         seg = df.iloc[segment*rows : segment*rows+rows]
#         x_raw = seg[feature]
#         x = x_raw.values
#         #y = seg['time_to_failure'].values[-1]
        
#         #y_tr.loc[segment, 'time_to_failure'] = y
#         df.loc[segment, f'{feature}_ave'] = x.mean()
#         # df.loc[segment, f'{feature}_std'] = x.std()
#         # df.loc[segment, f'{feature}_max'] = x.max()
#         # df.loc[segment, f'{feature}_min'] = x.min()
#         # df.loc[segment, f'{feature}_q01'] = np.quantile(x,0.01)
#         # df.loc[segment, f'{feature}_q05'] = np.quantile(x,0.05)
#         # df.loc[segment, f'{feature}_q95'] = np.quantile(x,0.95)
#         # df.loc[segment, f'{feature}_q99'] = np.quantile(x,0.99)
#         # df.loc[segment, f'{feature}_abs_median'] = np.median(np.abs(x))
#         # df.loc[segment, f'{feature}_abs_q95'] = np.quantile(np.abs(x),0.95)
#         # df.loc[segment, f'{feature}_abs_q99'] = np.quantile(np.abs(x),0.99)
#         # #df.loc[segment, 'F_test'], df.loc[segment, 'p_test'] = stats.f_oneway(x[:30000],x[30000:60000],x[60000:90000],x[90000:120000],x[120000:])
#         # df.loc[segment, f'{feature}_av_change_abs'] = np.mean(np.diff(x))
#         # df.loc[segment, f'{feature}_av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
#         # df.loc[segment, f'{feature}_abs_max'] = np.abs(x).max()
        
#         for windows in [125000, 250000]:
#             x_roll_std = x_raw.rolling(windows).std().dropna().values
#             x_roll_mean = x_raw.rolling(windows).mean().dropna().values
            
#             df.loc[segment, f'{feature}_ave_roll_std_' + str(windows)] = x_roll_std.mean()
#             # df.loc[segment, f'{feature}_std_roll_std_' + str(windows)] = x_roll_std.std()
#             # df.loc[segment, f'{feature}_max_roll_std_' + str(windows)] = x_roll_std.max()
#             # df.loc[segment, f'{feature}_min_roll_std_' + str(windows)] = x_roll_std.min()
#             # df.loc[segment, f'{feature}_q01_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.01)
#             # df.loc[segment, f'{feature}_q05_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.05)
#             # df.loc[segment, f'{feature}_q95_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.95)
#             # df.loc[segment, f'{feature}_q99_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.99)
#             # df.loc[segment, f'{feature}_av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
#             # df.loc[segment, f'{feature}_av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
#             # df.loc[segment, f'{feature}_abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
            
#             # df.loc[segment, f'{feature}_ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
#             # df.loc[segment, f'{feature}_std_roll_mean_' + str(windows)] = x_roll_mean.std()
#             # df.loc[segment, f'{feature}_max_roll_mean_' + str(windows)] = x_roll_mean.max()
#             # df.loc[segment, f'{feature}_min_roll_mean_' + str(windows)] = x_roll_mean.min()
#             # df.loc[segment, f'{feature}_q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.01)
#             # df.loc[segment, f'{feature}_q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.05)
#             # df.loc[segment, f'{feature}_q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.95)
#             # df.loc[segment, f'{feature}_q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.99)
#             # df.loc[segment, f'{feature}_av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
#             # df.loc[segment, f'{feature}_av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
#             # df.loc[segment, f'{feature}_abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

#     return df

def create_rolling_features(df, feature, rows):
    
    features_original = list(df.columns)
    
    #rolling features
    df[f'{feature}_avg_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).mean()
    df[f'{feature}_std_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).std()
    df[f'{feature}_max_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).max()
    df[f'{feature}_min_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).min()
    df[f'{feature}_q01_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).quantile(0.01)
    df[f'{feature}_q05_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).quantile(0.05)
    df[f'{feature}_q50_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).quantile(0.50)
    df[f'{feature}_q95_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).quantile(0.95)
    df[f'{feature}_q99_roll_' + str(rows)] = df[feature].rolling(window=rows, min_periods=1).quantile(0.99)
    df[f'{feature}_chg_roll_' + str(rows)] = ((df[feature] - df[feature].shift(rows))).fillna(0)
    df[f'{feature}_chg_rate_roll_' + str(rows)] = ((df[feature] - df[feature].shift(rows)) / df[feature].shift(rows)).fillna(0)
    #df[f'{feature}_trend_roll_' + str(rows)] = (df[feature].rolling(window=rows, min_periods=2).apply(calculate_slope, raw=True))  # Use a rolling window of n rows
    print(r"Initial rolling features are done...")
    
    for window in [1*37_000]:  # ~37_000 rows per day
        #rolling std from original rolling feature
        df[f'{feature}_avg_roll_std_' + str(window)] = df[f'{feature}_avg_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_std_roll_std_' + str(window)] = df[f'{feature}_std_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_max_roll_std_' + str(window)] = df[f'{feature}_max_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_min_roll_std_' + str(window)] = df[f'{feature}_min_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_q01_roll_std_' + str(window)] = df[f'{feature}_q01_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_q05_roll_std_' + str(window)] = df[f'{feature}_q05_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_q50_roll_std_' + str(window)] = df[f'{feature}_q50_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_q95_roll_std_' + str(window)] = df[f'{feature}_q95_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_chg_roll_std_' + str(window)] = df[f'{feature}_chg_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
        df[f'{feature}_chg_rate_roll_std_' + str(window)] = df[f'{feature}_chg_rate_roll_' + str(rows)].rolling(window=rows, min_periods=1).std()
                
        #rolling std from original rolling feature
        df[f'{feature}_avg_roll_avg_' + str(window)] = df[f'{feature}_avg_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_std_roll_avg_' + str(window)] = df[f'{feature}_std_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_max_roll_avg_' + str(window)] = df[f'{feature}_max_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_min_roll_avg_' + str(window)] = df[f'{feature}_min_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_q01_roll_avg_' + str(window)] = df[f'{feature}_q01_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_q05_roll_avg_' + str(window)] = df[f'{feature}_q05_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_q50_roll_avg_' + str(window)] = df[f'{feature}_q50_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_q95_roll_avg_' + str(window)] = df[f'{feature}_q95_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_chg_roll_avg_' + str(window)] = df[f'{feature}_chg_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        df[f'{feature}_chg_rate_roll_avg_' + str(window)] = df[f'{feature}_chg_rate_roll_' + str(rows)].rolling(window=rows, min_periods=1).mean()
        print(f"Addtional rolling features for window {window} are done..")
        
    features_all = list(df.columns)
    features_new = [col for col in features_all if col not in features_original]
    
    return df, features_new



# df_temp = df[['id','date_id','time_id','symbol_id','weight','feature_01','feature_02']]
# df_check = df[['id','date_id','time_id','symbol_id']]


#df, features_rolling = create_rolling_features(df, "feature_61", 37_000) # ~37_000 rows per day
#reduce_mem_usage(df, False)
# feature_names = feature_names + features_rolling


# addtional features 
# https://www.kaggle.com/code/yanisbelami/jane-street-real-time-market-data-forecasting-eda#Statistical-Tests
#df['feature_16_17_product'] = df['feature_16'] * df['feature_17']
df['feature_16_36_product'] = df['feature_16'] * df['feature_36']
df['responder_3_7_8_lag_1_avg'] = (df['responder_3_lag_1'] + df['responder_7_lag_1'] + df['responder_8_lag_1']) / 3
#df['responder_3_7_8_sum'] = df[['responder_3', 'responder_7', 'responder_8']].sum(axis=1)
df['feature_36_squared'] = df['feature_36'] ** 2
#df['feature_16_17_ratio'] = df['feature_16'] / (df['feature_17'] + 1e-9)
#df['feature_16_rolling_mean'] = df['feature_16'].rolling(window=5, min_periods=1).mean()
#df['feature_16_rolling_std'] = df['feature_16'].rolling(window=5, min_periods=1).std()

addtional_features = ['feature_36_squared','feature_16_36_product','responder_3_7_8_lag_1_avg']



# date_time features from yunsuxiaozi
# https://www.kaggle.com/code/yunsuxiaozi/js2024-starter
df['sin_time_id']=np.sin(2*np.pi*df['time_id']/967)
df['cos_time_id']=np.cos(2*np.pi*df['time_id']/967)
df['sin_time_id_halfday']=np.sin(2*np.pi*df['time_id']/483)
df['cos_time_id_halfday']=np.cos(2*np.pi*df['time_id']/483)
    
time_id_feature = ['sin_time_id','cos_time_id','sin_time_id_halfday','cos_time_id_halfday']



# override feature names to top 
feature_names_0 = [f"feature_{i:02d}" for i in range(79)]
feature_lagged_responders =  [f"responder_{idx}_lag_1" for idx in range(9)]
#feature_lagged_responders = ['responder_1_lag_1', 'responder_6_lag_1', 'responder_7_lag_1']
diff_features = ['diff_feature_15_feature_36','diff_feature_07_feature_60']
rolling_features = ['feature_61_chg_rate_roll_avg_37000','feature_61_chg_rate_roll_37000',
                    'feature_61_std_roll_std_37000','feature_61_chg_roll_avg_37000','feature_61_chg_rate_roll_std_37000']
# feature_names = feature_names_0 + feature_lagged_responders + diff_features + rolling_features
remove_features = ['feature_15', 'feature_17', 'feature_32', 'feature_33', 'feature_39', 'feature_41','feature_42', 
                   'feature_44', 'feature_50', 'feature_52', 'feature_53', 'feature_55', 'feature_58', 'feature_73', 'feature_74',
                   'feature_63', 'feature_54', 'feature_43']

feature_names = feature_names_0 + time_id_feature + feature_lagged_responders + addtional_features + diff_features
feature_names = [feature for feature in feature_names if feature not in remove_features]




#df = reduce_mem_usage(df, False)
# If in training mode, prepare validation data
# Extract features, target, and weights for validation dates
X_valid = df[feature_names].loc[df['date_id'].isin(valid_dates)]
y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)]
w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)]






    


# additional descriptive features
# def descriptive_stat(df, feature):
#     #features = df.columns.tolist()
#     features = feature
#     df['mean_features'] = df[features].mean(axis=1)
#     df['std_features'] = df[features].std(axis=1)
#     df['max_features'] = df[features].max(axis=1)
#     df['min_features'] = df[features].min(axis=1)
#     df['median_features'] = df[features].median(axis=1)
#     df['range_features'] = df['max_features'] - df['min_features']
#     df['90percentile_features'] = np.percentile(df[features], 90, axis=1)
#     df['75percentile_features'] = np.percentile(df[features], 75, axis=1)
#     df['25percentile_features'] = np.percentile(df[features], 25, axis=1)
#     df['10percentile_features'] = np.percentile(df[features], 10, axis=1)
#     df['kurtosis_features'] = scipy.stats.kurtosis(df[features], axis=1)
#     df['skew_features'] = scipy.stats.skew(df[features], axis=1)
    
#     mean_abs_dev = (df[features] - df[features].mean(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
#     median_abs_dev = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
#     range_abs_diff = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().max(axis=1) - (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().min(axis=1)
#     geometric_mean = np.exp(np.log(df[features].replace(0, 1)).mean(axis=1))
#     harmonic_mean = len(features) / (1 / df[features].replace(0, 1)).sum(axis=1)
#     coeff_variation = df['std_features'] / df['mean_features']
#     df['mean_absolute_deviation'] = mean_abs_dev
#     df['median_absolute_deviation'] = median_abs_dev
#     df['range_abs_diff'] = range_abs_diff
#     df['geometric_mean'] = geometric_mean
#     df['harmonic_mean'] = harmonic_mean
#     df['coeff_variation'] = coeff_variation
#     # just keep the descriptive statistics
#     #dataset = df.drop(features, axis=1)
    
#     return df



# # Initialize a list to store trained models
# models = []


# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return 'r2', r2, True


# Number of folds for cross-validation
N_fold = 1
#i = 0


# Select dates for training based on the fold number
i=0

selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_fold == i]

# # Specify model
# model =lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2')

# Extract features, target, and weights for the selected training dates
X_train = df[feature_names].loc[df['date_id'].isin(selected_dates)]
y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)]
w_train = df['weight'].loc[df['date_id'].isin(selected_dates)]



# fill nan with -1
X_train = X_train.fillna(-99)
X_valid = X_valid.fillna(-99)



del df
# Collect garbage to free up memory
import gc
gc.collect()




def objective(trial):
    # Define parameter search space
    param = {
        "objective": "regression",
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 400, 500]),
        #"metric": "None",  # Disable default metrics
        "metric": "rmse",
        "boosting_type": 'gbdt', #trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 128, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 0.8),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 0.8),
        "bagging_freq": trial.suggest_int("bagging_freq", 8, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 128, 256),
        "max_depth": trial.suggest_int("max_depth", 8, 32),  # -1 means no limit
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.001, 0.1),
        "device_type": "gpu",  # Enable GPU support
        "seed" : 12

    }

    # Create a LightGBM dataset
    dtrain = lgb.Dataset(X_train, y_train, weight=w_train)
    dval = lgb.Dataset(X_valid, y_valid, weight=w_valid, reference=dtrain)

    # Train LightGBM model
    model = lgb.train(
        param,
        dtrain,
        valid_sets=[dval],
        #feval=lambda y_pred, dval: r2_lgb(dval.get_label(), y_pred, dval.get_weight()),  # Use weights in the custom metric
        callbacks=[
            lgb.early_stopping(100), 
            lgb.log_evaluation(10)]
    )

    # Use the best score (maximized R²) as the objective to minimize (negative sign)
    best_score = model.best_score["valid_0"]["rmse"]
    return best_score
 

# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=3600*7) # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best rmse:", -study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = -study.best_value

# Format the file name with the best score
file_name = model_path + f"lgb_with_lag_add_diff_79_parameters_rmse_{best_score:.4f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")



# best_params = {
#     'n_estimators': 400,
#     'boosting_type': 'gbdt',
#     'num_leaves': 132,
#     'learning_rate': 0.01697482033838441,
#     'feature_fraction': 0.6766159288033143,
#     'bagging_fraction': 0.6640290696090753,
#     'bagging_freq': 9,
#     'min_data_in_leaf': 199,
#     'max_depth': 11,
#     'lambda_l1': 0.010969338788828302,
#     'lambda_l2': 0.017670611642700187}
#[400]	valid_0's l2: 0.643044	valid_0's r2: 0.00815552




# Function to train a model or load a pre-trained model
model_name = 'lgb_with_lag_add_diff_79_hyper'


# Train the model based on the type (LightGBM, XGBoost, or CatBoost)
model =lgb.LGBMRegressor(device='gpu', gpu_use_dp=True, objective='l2', **best_params) # from Hyper param tuning


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

# Output the best weighted r2 score
r2 = max(model.evals_result_['valid_0']['r2'])
print(f"valid r2: {r2}")

  
    
# Save the trained model to a file
joblib.dump(model, f'./models/{model_name}_{i}_{r2:.6f}.model')


# Save the feature names
#feature_names_pd = pd.DataFrame(feature_names)
#feature_names_pd.to_csv(f'./models/{model_name}_{i}_{r2:.6f}_feature_names.csv')






#assess the feature importance
lgb.plot_importance(model, max_num_features=25)  # Limit to top 30 features
plt.show()
    


# Create a DataFrame
lgb_feature_importance= pd.DataFrame({
    'Feature': model.feature_name_,
    'Importance': model.feature_importances_
})

lgb_feature_importance = lgb_feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
lgb_feature_importance.to_csv(model_path + f'{model_name}_0_{r2:.6f}.csv', index=False)









FIT_RIDGE = False
# Ridge regression
#if FIT_RIDGE:
ridge_features = feature_names

from sklearn.linear_model import Ridge


def custom_metric(y_true,y_pred,weight):
    weighted_r2=1-(np.sum(weight*(y_true-y_pred)**2)/np.sum(weight*y_true**2))
    return weighted_r2


X_train = X_train.fillna(-99)
X_valid = X_valid.fillna(-99)

results = []

for alpha in range(0,1000000,100000):
    #print(f'reg alpha is {alpha}')
    linear_model = Ridge(alpha)
    linear_model.fit(X_train[ridge_features],y_train)
    pred_train = linear_model.predict(X_train[ridge_features])
    pred_valid = linear_model.predict(X_valid[ridge_features])
    
    eps=1e-10  
    pred_valid=np.clip(pred_valid,-5+eps,5-eps)
    
    train_r2 = custom_metric(y_train,pred_train,weight=w_train)
    valid_r2 = custom_metric(y_valid,pred_valid,weight=w_valid)
    
    print(f'reg alpha is {alpha}:' + f"train weighted_r2:{train_r2:.12f}" + ' ' + f"test weighted_r2:{valid_r2:.12f}")
    #print(f"test weighted_r2:{valid_r2}")
    
    results.append({
    "alpha": alpha,
    "train_r2": train_r2,
    "valid_r2": valid_r2})
    
ridge_results = pd.DataFrame(results)

#fit the model with the best alpha
best_alpha = ridge_results.loc[ridge_results['valid_r2'].idxmax(), 'alpha']
linear_model = Ridge(best_alpha)
linear_model.fit(X_train[ridge_features],y_train)

#predict the valid set
pred_ridge = linear_model.predict(X_valid[ridge_features])
#eps=1e-10  
#pred_valid=np.clip(pred_valid,-5+eps,5-eps)

#valid_r2 = custom_metric(y_valid,pred_valid,weight=w_valid)



FIT_ENSEMBLE = False

#if FIT_ENSEMBLE:
# Load lgb model and predictions
lgb_model = joblib.load(r'C:\\Users\\zrj-desktop\\models\\lgb_with_lag_add_77_hyper_0_0.008203.model')
pred_lgb = lgb_model.predict(X_valid.to_numpy())
r2_lgb(pred_lgb,y_valid,w_valid)


# Find the optimal weight
for w in np.arange(0, 1.25, 0.25):
    pred_ensemble = w*pred_ridge + (1-w)*pred_lgb
    
    eps=1e-10  
    pred_ensemble=np.clip(pred_ensemble,-5+eps,5-eps)
    valid_r2 = custom_metric(pred_ensemble,pred_valid,weight=w_valid)
    
    print(f'weight {w} has weighted r2 of {valid_r2}')