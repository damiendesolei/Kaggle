# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:58:18 2025

@author: zrj-desktop
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib 
#import itertools

import pandas as pd
pd.set_option('display.max_columns', None)
import polars as pl
import numpy as np 
#import stats
import scipy

from sklearn.linear_model import LinearRegression


import lightgbm as lgb
#from sklearn.metrics import r2_score
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
PATH = '/kaggle/input/rohlik-sales-forecasting-challenge' if os.path.exists('/kaggle/input/rohlik-sales-forecasting-challenge') else r'G:\\kaggle\Rohlik_Sales_Forecasting_Challenge\\'


# list all files in the path
for dirname, _, filenames in os.walk(PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        

# Read in the files
sales_train = pd.read_csv(PATH + 'sales_train.csv', parse_dates=['date'])
sales_test = pd.read_csv(PATH + 'sales_test.csv', parse_dates=['date'])
inventory = pd.read_csv(PATH + 'inventory.csv')
calendar = pd.read_csv(PATH + 'calendar.csv', parse_dates=['date'])


# Join with Inventory
sales_train = pd.merge(sales_train, inventory, how='left', on =['unique_id','warehouse'])
sales_test = pd.merge(sales_test, inventory, how='left', on =['unique_id','warehouse'])


# Join with Calendar
sales_train = pd.merge(sales_train, calendar, how='left', on =['date','warehouse'])
sales_test = pd.merge(sales_test, calendar, how='left', on =['date','warehouse'])


# check column difference
np.setdiff1d(sales_train.columns,sales_test.columns)




# date related features
def add_date_features(df):
    df['year'] = df['date'].dt.year
    #df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    
    
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_dayofyear']=np.sin(2*np.pi*df['dayofyear']/365)
    df['cos_dayofyear']=np.cos(2*np.pi*df['dayofyear']/365)
    
    df['dayofweek']=df['date'].dt.dayofweek
    #df['weekday'] = df['date'].dt.weekday
    df['weekend']=(df['dayofweek']>4).astype(np.int8)
    df['sin_dayofweek']=np.sin(2*np.pi*df['dayofweek']/7)
    df['cos_dayofweek']=np.cos(2*np.pi*df['dayofweek']/7)
    
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['sin_weekofyear']=np.sin(2*np.pi*df['weekofyear']/52)
    df['cos_weekofyear']=np.cos(2*np.pi*df['weekofyear']/52)
    
    df['quarter']=df['date'].dt.quarter
    df['sin_quarter']=np.sin(2*np.pi*df['quarter']/4)
    df['cos_quarter']=np.cos(2*np.pi*df['quarter']/4)
    
    df['month']=df['date'].dt.month
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['sin_month']=np.sin(2*np.pi*df['month']/12)
    df['cos_month']=np.cos(2*np.pi*df['month']/12)
    
    
    df['date_copy'] = df['date']
    return df


sales_train = add_date_features(sales_train)


date_columns = [col for col in sales_train.columns if 'date' in col]
check = sales_train[date_columns]