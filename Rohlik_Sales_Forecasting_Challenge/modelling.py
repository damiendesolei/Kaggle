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
from datetime import datetime, timedelta

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
train = pd.read_csv(PATH + 'sales_train.csv', parse_dates=['date'])
test = pd.read_csv(PATH + 'sales_test.csv', parse_dates=['date'])
print(f'original train df dimention is {train.shape}')

test_id = test['unique_id'].unique() #only use unique_id in testset
train = train[train['unique_id'].isin(test_id)]
print(f'modified train df dimention is {train.shape}')


# Read in other files
inventory = pd.read_csv(PATH + 'inventory.csv')
calendar = pd.read_csv(PATH + 'calendar.csv', parse_dates=['date'])


# Join with Inventory
train = train.merge(inventory, how='left', on =['unique_id','warehouse'])
test = test.merge(inventory, how='left', on =['unique_id','warehouse'])


# Join with Calendar
train = train.merge(calendar, how='left', on =['date','warehouse'])
test = test.merge(calendar, how='left', on =['date','warehouse'])


# check column difference
np.setdiff1d(train.columns, test.columns)




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


train = add_date_features(train)



# Holidays 
# https://www.kaggle.com/code/yunsuxiaozi/rsfc-yunbase
holidays_prague = [ 
    (['01/01/2020', '01/01/2021', '01/01/2022', '01/01/2023', '01/01/2024'], 'New Years Day'),
    (['04/10/2020', '04/02/2021', '04/15/2022', '04/07/2023', '03/29/2024'], 'Good Friday'),
    (['04/13/2020', '04/05/2021', '04/18/2022', '04/10/2023', '04/01/2024'], 'Easter Monday'),
    (['05/01/2020', '05/01/2021', '05/01/2022', '05/01/2023', '05/01/2024'], 'Labour Day'),
    (['05/08/2020', '05/08/2021', '05/08/2022', '05/08/2023', '05/08/2024'], 'Liberation Day'),
    (['07/05/2020', '07/05/2021', '07/05/2022', '07/05/2023', '07/05/2024'], 'St. Cyril and Methodius Day'),
    (['07/06/2020', '07/06/2021', '07/06/2022', '07/06/2023', '07/06/2024'], 'Jan Hus Day'),
    (['09/28/2020', '09/28/2021', '09/28/2022', '09/28/2023', '09/28/2024'], 'Statehood Day'),
    (['10/28/2020', '10/28/2021', '10/28/2022', '10/28/2023', '10/28/2024'], 'Independent Czechoslovak State Day'),
    (['11/17/2020', '11/17/2021', '11/17/2022', '11/17/2023', '11/17/2024'], 'Struggle for Freedom and Democracy Day'),
    (['12/24/2020', '12/24/2021', '12/24/2022', '12/24/2023', '12/24/2024'], 'Christmas Eve'),
    (['12/25/2020', '12/25/2021', '12/25/2022', '12/25/2023', '12/25/2024'], 'Christmas Day'),
    (['12/26/2020', '12/26/2021', '12/26/2022', '12/26/2023', '12/26/2024'], 'St. Stephens Day'),
]


holidays_brno = [
    (['01/01/2020', '01/01/2021', '01/01/2022', '01/01/2023', '01/01/2024'], 'New Years Day'),
    (['04/10/2020', '04/02/2021', '04/15/2022', '04/07/2023', '03/29/2024'], 'Good Friday'),
    (['04/13/2020', '04/05/2021', '04/18/2022', '04/10/2023', '04/01/2024'], 'Easter Monday'),
    (['05/01/2020', '05/01/2021', '05/01/2022', '05/01/2023', '05/01/2024'], 'Labour Day'),
    (['05/08/2020', '05/08/2021', '05/08/2022', '05/08/2023', '05/08/2024'], 'Liberation Day'),
    (['07/05/2020', '07/05/2021', '07/05/2022', '07/05/2023', '07/05/2024'], 'St. Cyril and Methodius Day'),
    (['07/06/2020', '07/06/2021', '07/06/2022', '07/06/2023', '07/06/2024'], 'Jan Hus Day'),
    (['09/28/2020', '09/28/2021', '09/28/2022', '09/28/2023', '09/28/2024'], 'Statehood Day'),
    (['10/28/2020', '10/28/2021', '10/28/2022', '10/28/2023', '10/28/2024'], 'Independent Czechoslovak State Day'),
    (['11/17/2020', '11/17/2021', '11/17/2022', '11/17/2023', '11/17/2024'], 'Struggle for Freedom and Democracy Day'),
    (['12/24/2020', '12/24/2021', '12/24/2022', '12/24/2023', '12/24/2024'], 'Christmas Eve'),
    (['12/25/2020', '12/25/2021', '12/25/2022', '12/25/2023', '12/25/2024'], 'Christmas Day'),
    (['12/26/2020', '12/26/2021', '12/26/2022', '12/26/2023', '12/26/2024'], 'St. Stephens Day'),
]


holidays_budapest = [
    (['01/01/2020', '01/01/2021', '01/01/2022', '01/01/2023', '01/01/2024'], 'New Years Day'),
    (['03/15/2020', '03/15/2021', '03/15/2022', '03/15/2023', '03/15/2024'], 'National Day (1848 Revolution Memorial)'),
    (['04/10/2020', '04/02/2021', '04/15/2022', '04/07/2023', '03/29/2024'], 'Good Friday'),
    (['04/13/2020', '04/05/2021', '04/18/2022', '04/10/2023', '04/01/2024'], 'Easter Monday'),
    (['05/01/2020', '05/01/2021', '05/01/2022', '05/01/2023', '05/01/2024'], 'Labour Day'),
    (['06/01/2020', '05/24/2021', '06/06/2022', '05/29/2023', '05/20/2024'], 'Whit Monday'),
    (['08/20/2020', '08/20/2021', '08/20/2022', '08/20/2023', '08/20/2024'], 'St. Stephens Day'),
    (['10/23/2020', '10/23/2021', '10/23/2022', '10/23/2023', '10/23/2024'], 'Republic Day'),
    (['11/01/2020', '11/01/2021', '11/01/2022', '11/01/2023', '11/01/2024'], 'All Saints Day'),
    (['12/25/2020', '12/25/2021', '12/25/2022', '12/25/2023', '12/25/2024'], 'Christmas Day'),
    (['12/26/2020', '12/26/2021', '12/26/2022', '12/26/2023', '12/26/2024'], 'Second Day of Christmas')
]
   
  
holidays_munich = [
    (['01/01/2020', '01/01/2021', '01/01/2022', '01/01/2023', '01/01/2024'], 'New Years Day'),
    (['01/06/2020', '01/06/2021', '01/06/2022', '01/06/2023', '01/06/2024'], 'Epiphany'),
    (['04/10/2020', '04/02/2021', '04/15/2022', '04/07/2023', '03/29/2024'], 'Good Friday'),
    (['04/13/2020', '04/05/2021', '04/18/2022', '04/10/2023', '04/01/2024'], 'Easter Monday'),
    (['05/01/2020', '05/01/2021', '05/01/2022', '05/01/2023', '05/01/2024'], 'Labour Day'),
    (['05/21/2020', '05/13/2021', '05/26/2022', '05/18/2023', '05/09/2024'], 'Ascension Day'),
    (['06/01/2020', '05/24/2021', '06/06/2022', '05/29/2023', '05/20/2024'], 'Whit Monday'),
    (['06/11/2020', '06/03/2021', '06/16/2022', '06/08/2023', '05/30/2024'], 'Corpus Christi'),
    (['08/15/2020', '08/15/2021', '08/15/2022', '08/15/2023', '08/15/2024'], 'Assumption Day'),
    (['10/03/2020', '10/03/2021', '10/03/2022', '10/03/2023', '10/03/2024'], 'German Unity Day'),
    (['11/01/2020', '11/01/2021', '11/01/2022', '11/01/2023', '11/01/2024'], 'All Saints Day'),
    (['12/25/2020', '12/25/2021', '12/25/2022', '12/25/2023', '12/25/2024'], 'Christmas Day'),
    (['12/26/2020', '12/26/2021', '12/26/2022', '12/26/2023', '12/26/2024'], 'St. Stephens Day'),
]


holidays_frankfurt = [
    (['01/01/2020', '01/01/2021', '01/01/2022', '01/01/2023', '01/01/2024'], 'New Years Day'),
    (['01/06/2020', '01/06/2021', '01/06/2022', '01/06/2023', '01/06/2024'], 'Epiphany'),
    (['04/10/2020', '04/02/2021', '04/15/2022', '04/07/2023', '03/29/2024'], 'Good Friday'),
    (['04/13/2020', '04/05/2021', '04/18/2022', '04/10/2023', '04/01/2024'], 'Easter Monday'),
    (['05/01/2020', '05/01/2021', '05/01/2022', '05/01/2023', '05/01/2024'], 'Labour Day'),
    (['05/21/2020', '05/13/2021', '05/26/2022', '05/18/2023', '05/09/2024'], 'Ascension Day'),
    (['06/01/2020', '05/24/2021', '06/06/2022', '05/29/2023', '05/20/2024'], 'Whit Monday'),
    (['06/11/2020', '06/03/2021', '06/16/2022', '06/08/2023', '05/30/2024'], 'Corpus Christi'),
    (['08/15/2020', '08/15/2021', '08/15/2022', '08/15/2023', '08/15/2024'], 'Assumption Day'),
    (['10/03/2020', '10/03/2021', '10/03/2022', '10/03/2023', '10/03/2024'], 'German Unity Day'),
    (['11/01/2020', '11/01/2021', '11/01/2022', '11/01/2023', '11/01/2024'], 'All Saints Day'),
    (['12/25/2020', '12/25/2021', '12/25/2022', '12/25/2023', '12/25/2024'], 'Christmas Day'),
    (['12/26/2020', '12/26/2021', '12/26/2022', '12/26/2023', '12/26/2024'], 'St. Stephens Day'),

]


def fill_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    
    #add features
    df['long_weekend'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(np.int8)
    
    return df


train = fill_holidays(df_fill=train, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=holidays_prague)
train = fill_holidays(df_fill=train, warehouses=['Brno_1'], holidays=holidays_brno)
train = fill_holidays(df_fill=train, warehouses=['Munich_1'], holidays=holidays_munich)
train = fill_holidays(df_fill=train, warehouses=['Frankfurt_1'], holidays=holidays_frankfurt)
train = fill_holidays(df_fill=train, warehouses=['Budapest_1'], holidays=holidays_budapest)
print(f"train.shape:{train.shape}")




# Find all weekend dates
start_date = train['date'].min()
end_date = train['date'].max()
#start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
#end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
current_date = start_date
weekends = []
while current_date <= end_date:
    if current_date.weekday() == 5 or current_date.weekday() == 6:
        weekends.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)