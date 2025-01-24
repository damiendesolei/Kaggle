# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:58:18 2025

@author: zrj-desktop
"""

import warnings
warnings.filterwarnings("ignore")

import os
import datetime as dt

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
from sklearn.model_selection import TimeSeriesSplit, train_test_split

import lightgbm as lgb
#from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


from tqdm import tqdm, tqdm_notebook
from joblib import Parallel, delayed


import optuna



model_path = r'G:\\kaggle\Rohlik_Sales_Forecasting_Challenge\model\\'


def reduce_mem_usage(df, float16_as32=True):
    #memory_usage()是df每列的内存使用量,sum是对它们求和, B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of current dataframe is {:.2f} MB'.format(start_mem))
    non_date_columns = [col for col in df.columns if df[col].dtype != 'datetime64[ns]']

    for col in non_date_columns:#遍历每列的列名
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
    #current ram
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #ram reduction%
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



# Define the path to the input data directory
# If the local directory exists, use it; otherwise, use the Kaggle input directory
PATH = '/kaggle/input/rohlik-sales-forecasting-challenge' if os.path.exists('/kaggle/input/rohlik-sales-forecasting-challenge') else r'G:\\kaggle\Rohlik_Sales_Forecasting_Challenge\\'


# list all files in the path
# for dirname, _, filenames in os.walk(PATH):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
        

# Read in the files
train = pd.read_csv(PATH + 'sales_train.csv', parse_dates=['date'])
test = pd.read_csv(PATH + 'sales_test.csv', parse_dates=['date'])
print(f'{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} original train df dimention is {train.shape}')

test_id = test['unique_id'].unique() #only use unique_id in testset
train = train[train['unique_id'].isin(test_id)]
#train = train[train.date>='2021-01-01 00:00:00'] # only use post-covid data
print(f'{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} filtered train df dimention is {train.shape}')


# Read in other files
inventory = pd.read_csv(PATH + 'inventory.csv')
calendar = pd.read_csv(PATH + 'calendar.csv', parse_dates=['date'])
weights = pd.read_csv(PATH + 'test_weights.csv') 


# Join with Inventory
train = train.merge(inventory, how='left', on =['unique_id','warehouse'])
test = test.merge(inventory, how='left', on =['unique_id','warehouse'])


# Join with Calendar
train = train.merge(calendar, how='left', on =['date','warehouse'])
test = test.merge(calendar, how='left', on =['date','warehouse'])


# Join with Weight
train = train.merge(weights, how='left', on =['unique_id'])
test = test.merge(weights, how='left', on =['unique_id'])


# check column difference
print(f'{np.setdiff1d(train.columns, test.columns)} in train, but not in test')


# combine train and test
total = pd.concat((train, test))


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
    
    df['day']=df['date'].dt.day
    df['dayofmonth']=df['day']//10
    df['sin_day']=np.sin(2*np.pi*df['day']/30)
    df['cos_day']=np.cos(2*np.pi*df['day']/30)
    
    return df


total = add_date_features(total)



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
    #df['long_weekend'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(np.int8)
    
    return df

# check = train[['date','warehouse','holiday_name','holiday','shops_closed','winter_school_holidays','school_holidays']]
# check = check[train['holiday_name'].notna()]
total = fill_holidays(total, ['Prague_1', 'Prague_2', 'Prague_3'], holidays_prague)
total = fill_holidays(total, ['Brno_1'], holidays_brno)
total = fill_holidays(total, ['Munich_1'], holidays_munich)
total = fill_holidays(total, ['Frankfurt_1'], holidays_frankfurt)
total = fill_holidays(total, ['Budapest_1'], holidays_budapest)
print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} total.shape:{total.shape}")





# Find all weekend dates - to flag out weekend
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
    
    
    
    
# Food Types
def get_food_type(food):
    food_types = {
        
        "fruit": [
            "Apple", "Avocado", "Banana", "Cucumber", "Lemon", "Mango", "Melon", 
            "Orange", "Pear", "Pineapple", "Pomegranate", "Grape", "Watermelon", 
            "Blueberry", "Lime", "Zucchini", "Grapefruit", "Physalis", "Berry", 
            "Tangerine", "Apricot", "Pomelo", "Blackberry", "Cherry", "Raspberry", 
            "Passion fruit", "Date", "Plum", "Fig", "Cactus Fruit", "Peach", 
            "Nectarine", "Strawberry", "Mandarin", "Persimmon", "Canteloupe", 
            "Lamb's lettuce"],
        
        "vegetable": [
            "Tomato", "Potato", "Mushroom", "Onion", "Lettuce", "Cabbage", "Carrot", 
            "Pepper", "Bell Pepper", "Radish", "Pumpkin", "Broccoli", "Basil", 
            "Cauliflower", "Leek", "Chive", "Eggplant", "Kohlrabi", "Asparagus", 
            "Rosemary", "Mint", "Chicory", "Fennel", "Strawberry", "Raspberry", 
            "Ginger", "Pak choi", "Green Bean", "Cress", "Pea", "Pomelo", "Chili", 
            "Squash", "Paprika", "Nut", "Plantain", "Soybean sprout", "Cantaloupe"],
        
        "meat": [
            "Chicken", "Pork", "Beef", "Turkey", "Mix meat", "Duck", "Plant meat",
            "Burger"],
        
        "fish": [
            "Salmon", "Shrimp", "Surimi"],
        
        "other": [
            "Herb", "Salad", "Parsley", "Garlic", "Beet", "Spinach", "Sweet Potato", 
            "Thyme", "Snack", "Arugula", "Grapefruit", "Physalis", "Berry", 
            "Shallot", "Corn", "Sprout", "Bean", "Cauliflower", "Leek", "Chive", 
            "Eggplant", "Kohlrabi", "Asparagus", "Rosemary", "Mint", "Chicory", 
            "Peach", "Nectarine", "Thyme", "Fennel", "Strawberry", "Raspberry", 
            "Ginger", "Passion fruit", "Date", "Plum", "Fig", "Bell pepper", 
            "Soup", "Cactus Fruit", "Pak choi", "Drink", "Pappudia", "Tangerine", 
            "Apricot", "Pea", "Pomelo", "Bag", "Chili", "Blackberry", "Granadilla", 
            "Cherry", "Squash", "Paprika", "Nut", "Plantain", "Mandarin", 
            "Soybean sprout", "Soil", "Cantaloupe", "Green Bean", "Persimmon", 
            "Cress", "Pepperoni", "Gooseberry", "Currant", "Flower"],
        
         "Bakery":[
           "Bread", "Pastry", "Roll", "Baguette", "Toust", "Croissant", "Tortilla",
           "Donut", "Snack", "Cake", "Pretzel", "Cracker", "Muffin", "Bagel",
           "Breadcrumb", "Pita", "Rice Cake", "Bun", "Waffle", "Biscuit",
           "Sandwich", "Cheese", "Wrap", "Breadcrumbs", "Focaccia", "Cookie",
           "Cream", "Cornmeal", "Dessert", "Grain", "Hot Dog", "Pasta", "Pizza",
           "Flatbread", "Yogurt", "Bakery", "Lucki", "Brioche"]
    }
    
    for food_type, food_list in food_types.items():
        if food in food_list:
            return food_type
    return 'other'


# Feature Engineering
def feature_engineering(df):
    
    df['index']=np.arange(len(df))
    df=df.sort_values(['date']).reset_index(drop=True)


    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} add autoregression feature >>>")
    for gap in [14, 20, 28, 35, 356, 364, 370]:
    #for gap in [14, 356]:
        df[f'sales_shift{gap}'] = df.groupby(['warehouse','name'])['sales'].shift(gap)
    

    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} extract names >>>")
    #name:'Pastry_196'
    df['name_0']=df['name'].apply(lambda x:x.split("_")[0])
    df['name_1']=df['name'].apply(lambda x:x.split("_")[1])
    df.drop(['name'],axis=1,inplace=True)
    for i in range(2,5): #strip out the numeric suffix
        df[f'L{i}_category_name_en']=df[f'L{i}_category_name_en'].apply(lambda x:x.split('_')[2])


    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} store2country feature >>>")
    store2country = {
        'Budapest_1': 'Hungary',
        'Prague_2': 'Czechia',
        'Brno_1': 'Czechia',
        'Prague_1': 'Czechia',
        'Prague_3': 'Czechia',
        'Munich_1': 'Germany',
        'Frankfurt_1': 'Germany'
    }
    df['country']=df['warehouse'].apply(lambda x:store2country[x])


    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} get food type >>>")
    df['L5_category_name_en']=df['name_0'].apply(lambda x:get_food_type(x))


    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} add weekend feature >>>")
    df.loc[(df['holiday_name'].isna())&(df['date'].isin(weekends)),'holiday_name']='weekend'
    #simple weekend
    df['is_holiday']=(df['holiday_name']==df['holiday_name']).astype(np.int8)
    #holiday but not weekend
    df.loc[(df['is_holiday']==1)&(df['holiday_name']!='weekend'),'is_holiday']=2
    
    
    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} get total discount >>>")
    df['total_type_discount']=0
    for i in range(7):
        df['total_type_discount']+=df[f'type_{i}_discount']
    
    df['dollar_discount'] = df['total_type_discount'] * df['sell_price_main']


    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} time diff and shift feature >>>")
    for gap in [1, 2]:
        for col in ['is_holiday','weekend']:
            df[col+f"_shift{gap}"]=df.groupby(['warehouse','unique_id','product_unique_id'])[col].shift(gap)

    # for col in ['total_orders','sell_price_main','total_type_discount']:#'total_orders*sell_price_main'
    #     for agg in ['std','skew','max']:#,'median']:
    #         df[f'{agg}_{col}_each_name_WU_per_day']=df.groupby(['date','warehouse','unique_id','name_0','name_1'])[col].transform(agg)
    #         df[f'{agg}_{col}_each_name0_WU_per_day']=df.groupby(['date','warehouse','unique_id','name_0'])[col].transform(agg)
    #         df[f'{agg}_{col}_each_L1_WU_per_day']=df.groupby(['date','warehouse','unique_id','L1_category_name_en'])[col].transform(agg)
    #         df[f'{agg}_{col}_each_name0_W_per_day']=df.groupby(['date','warehouse','name_0'])[col].transform(agg)
    #         df[f'{agg}_{col}_each_name0_per_day']=df.groupby(['date','name_0'])[col].transform(agg)
            
    #         for gap in [1]:
    #             df[f'{agg}_{col}_each_name_WU_per_day_diff{gap}']=df.groupby(['warehouse','unique_id','name_0','name_1'])[f'{agg}_{col}_each_name_WU_per_day'].diff(gap)
    #             df[f'{agg}_{col}_each_name0_WU_per_day_diff{gap}']=df.groupby(['warehouse','unique_id','name_0','name_1'])[f'{agg}_{col}_each_name0_WU_per_day'].diff(gap)
    #             df[f'{agg}_{col}_each_L1_WU_per_day_diff{gap}']=df.groupby(['warehouse','unique_id','name_0','name_1'])[f'{agg}_{col}_each_L1_WU_per_day'].diff(gap)
    #             df[f'{agg}_{col}_each_name0_W_per_day_diff{gap}']=df.groupby(['warehouse','unique_id','name_0','name_1'])[f'{agg}_{col}_each_name0_W_per_day'].diff(gap)
    #             df[f'{agg}_{col}_each_name0_per_day_diff{gap}']=df.groupby(['warehouse','unique_id','name_0','name_1'])[f'{agg}_{col}_each_name0_per_day'].diff(gap)
  
                
    df=df.sort_values(['index']).reset_index(drop=True)
    df.drop(['index'],axis=1,inplace=True)
    
    return df

total = feature_engineering(total)



# https://www.kaggle.com/code/darkswordmg/rohlik-2024-2nd-place-solution-single-lgbm?scriptVersionId=194105779
# Create 2 new columns "day_before_holiday" and "day_after_holiday"
total['day_before_holiday'] = total['holiday'].shift(-1).fillna(0)
total['day_after_holiday'] = total['holiday'].shift().fillna(0)
total['day_before_holiday'] = total['day_before_holiday'].astype(int)
total['day_after_holiday'] = total['day_after_holiday'].astype(int)



# Perform target encoding 
def target_encoding(df, cat_cols, target_variable, weight=10): # weight=0 -> no smooth

    for col in cat_cols:
        weight = weight
        feat = df.groupby(col)[target_variable].agg(["mean", "count"])
        mean = feat['mean']
        count = feat['count']
        
        smooth = (count * mean + weight * mean) / (weight + count)

        df.loc[:, col] = df.loc[:, col].map(smooth)

    return df

total = target_encoding(total, ['product_unique_id','unique_id','warehouse'], 'sales')






# Set up random column
def add_random_column(df):
    np.random.seed(24)
    df['random'] = np.random.rand(df.shape[0])
    return df

add_random_column(total)
total = reduce_mem_usage(total)


# Split the train and test
drop_cols=['availability']
total.drop([col for col in total.columns if total[col].isna().mean()>0.98]+drop_cols, axis=1, inplace=True)
#total.drop(drop_cols, axis=1, inplace=True)
train = total[:len(train)]
test = total[len(train):].drop(['sales'], axis=1)

# fill nan with -99
train = train.fillna(-99)
test = test.fillna(-99)

# check objective columns
# object_columns = [col for col in test.columns if test[col].dtype == 'object']
# print(test[object_columns].head())


print(f"train.shape:{train.shape}, test.shape:{test.shape}")
train.head()




# intial features with only numeric columns
features_0 = [col for col in test.columns if (test[col].dtype != 'object' and test[col].dtype != 'datetime64[ns]')]

# remove features based on previous model importance
#previous_features = pd.read_csv(model_path + "lgb_with_random_26_parameters_features_24.8880.csv")
#remove_features = list(previous_features[(previous_features.Importance<=5200)]['Feature'])

remove_features = ['weight'] #+ remove_features


features = [feature for feature in features_0 if feature not in remove_features]
#features = list(previous_features[(previous_features.Importance>=2000)&(previous_features.Feature!='random')]['Feature'])





# Setup model name to tune and predict
model_name = f'lgb_with_random_{len(features)}_parameters'


# define weighted MAE
def weighted_mae(y_true, y_pred, weight):
    wmae = np.sum(weight * np.abs(y_true-y_pred)) / np.sum(weight)
    return 'wmae', wmae, False # True = higher is better


#  Train test split
X = train[features]
y = train['sales']
w = train['weight']

X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(X, y, w, test_size=0.25, random_state=2025)
# X_train = train[features].loc[~X['month'].isin([6])]
# y_train = train['sales'].loc[~X['month'].isin([6])]
# w_train = train['weight'].loc[~X['month'].isin([6])]

# X_valid = train[features].loc[X['month'].isin([6])]
# y_valid = train['sales'].loc[X['month'].isin([6])]
# w_valid = train['weight'].loc[X['month'].isin([6])]



# Define the parameter space
def objective(trial):
    param = {
        'objective': 'regression',  
        'metric': 'mae',  
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 300, 500, step=100),
        'max_depth': trial.suggest_int('max_depth', 1, 32, step=2),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  
        'num_leaves': trial.suggest_int('num_leaves', 12, 256, step=2), 
        #'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),  
        #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),  
        'feature_fraction': trial.suggest_categorical('feature_fraction', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.001, 0.1),
        "device_type": "gpu",  
        "seed" : 2025

    }

    # Create a LightGBM dataset
    dtrain = lgb.Dataset(X_train, y_train, weight=w_train)
    dval = lgb.Dataset(X_valid, y_valid, weight=w_valid, reference=dtrain)

    # Train LightGBM model
    model = lgb.train(param,
        dtrain,
        valid_sets=[dval],
        #feval=lambda y_pred, dval: r2_lgb(dval.get_label(), y_pred, dval.get_weight()),  # Use weights in the custom metric
        callbacks=[
            lgb.early_stopping(100), 
            lgb.log_evaluation(10)]
    )

    # Use the best score (maximized R²) as the objective to minimize (negative sign)
    #print(model.best_score["valid_0"])
    best_score = model.best_score["valid_0"]["l1"]
    return best_score
    
    y_pred = model.predict(X_valid)
    wmae = weighted_mae(y_valid, y_pred, w_valid)  # WMAE for regression
    
    return wmae


# Run Optuna study
print("Start running hyper parameter tuning..")
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=3600*2) # 3600*n hour

# Print the best hyperparameters and score
print("Best hyperparameters:", study.best_params)
print("Best mae:", study.best_value)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Format the file name with the best score
file_name = model_path + model_name + f"_mae_{best_score:.4f}.csv"

# Save the best parameters to a CSV file
df_param = pd.DataFrame([best_params])  # Convert to DataFrame
df_param.to_csv(file_name, index=False)  # Save to CSV

print(f"Best parameters saved to {file_name}")



best_params = {'n_estimators': 500, 
               'max_depth': 17, 
               'learning_rate': 0.09984374689282284, 
               'num_leaves': 256, 
               'feature_fraction': 0.9, 
               'bagging_fraction': 1.0, 
               'bagging_freq': 10, 
               'lambda_l1': 0.046129269846374735, 
               'lambda_l2': 0.07193361763345359}
# Best mae: 16.389738883798906

# Model fitting and prediction
model =lgb.LGBMRegressor(device='gpu', gpu_use_dp=True, objective='l1', **best_params) # from Hyper param tuning


# weighted mae for lgb - weight will not be passed by lgb directly.
def weighted_mae_val(y_true, y_pred, sample_weight):
    wmae = np.average(np.abs(y_true-y_pred), weights=sample_weight)
    return 'wmae', wmae, False # True = higher is better


# Train LightGBM model with early stopping and evaluation logging
model.fit(X_train, y_train, w_train,  
          eval_metric=[weighted_mae_val],
          eval_set=[(X_valid, y_valid, w_valid)], 
          callbacks=[
              lgb.early_stopping(100), 
              lgb.log_evaluation(10)
          ])


# Append the trained model to the list
#models.append(model)

# Output the best weighted MAE
wmae = min(model.evals_result_['valid_0']['wmae'])
print(f"valid wmae: {wmae}")
#valid_0's l1: 28.02	valid_0's wmae: 28.02
  
    
# Save the trained model to a file
joblib.dump(model, model_path + f'{model_name}_wmae_{wmae:.4f}.model')






#assess the feature importance
lgb.plot_importance(model, max_num_features=25)  # Limit to top 30 features
plt.show()
    


# Create a DataFrame
lgb_feature_importance= pd.DataFrame({
    'Feature': model.feature_name_,
    'Importance': model.feature_importances_
})

lgb_feature_importance = lgb_feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
lgb_feature_importance.to_csv(model_path + f'{model_name}_features_{wmae:.4f}.csv', index=False)






# Predict and submit
test['sales_hat'] = model.predict(test[features])
test.loc[test['sales_hat'] < 0, 'sales_hat'] = 0

# Create id
test['id'] = test['unique_id'].astype(str) + "_" + test['date'].astype(str)
submission = test[['id','sales_hat']]
#submission.to_csv(model_path + f"{model_name}_submission_{wmae:.4f}.csv",index=False)


# correct id
CORRET = True
if CORRET:
    test_0 = pd.read_csv(PATH + 'sales_test.csv', parse_dates=['date'])
    test_0['id'] = test_0['unique_id'].astype(str) + "_" + test_0['date'].astype(str)
    submission = submission.reset_index(drop=True)
    submission['id'] = test_0['id']
    submission.to_csv(model_path + f"{model_name}_submission_{wmae:.4f}.csv",index=False)