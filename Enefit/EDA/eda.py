# -*- coding: utf-8 -*-


######Nov-11-2023######
####Shanghai-China#####

import pandas as pd 
pd.set_option('display.max_columns', None)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import holidays

from pathlib import Path

data_path = Path('C:/Users/damie/Downloads/predict-energy-behavior-of-prosumers')

train = pd.read_csv(data_path / 'train.csv', parse_dates=['datetime'])
train.info(memory_usage='deep')
train.dtypes
train.head()

# for col in train.columns:
#     print(str(col) + ' has unique values of:')
#     print(set(train[col].unique()))
    
    
historical_weather = pd.read_csv(data_path / 'historical_weather.csv', parse_dates=['datetime'])
historical_weather.info(memory_usage='deep')
historical_weather.dtypes
historical_weather.head()

# for col in historical_weather.columns:
#     print(str(col) + ' has unique values of:')
#     print(set(historical_weather[col].unique()))

# consumption = train[train["is_consumption"]==1]
# monthlyCons = consumption.groupby(pd.Grouper(key="datetime", freq='M')).mean()
    