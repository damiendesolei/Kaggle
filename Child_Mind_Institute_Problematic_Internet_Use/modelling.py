# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:24:10 2024

@author: damie
"""



import numpy as np
import pandas as pd


import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import *
from sklearn.metrics import *


from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor





#https://www.kaggle.com/code/abdmental01/cmi-best-single-model
def process_file(dirname, filename):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(dirname, fname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df


#load the data

#PATH = r'C:\Users\damie\Downloads\child-mind-institute-problematic-internet-use\\'
PATH = r'G:\kaggle\child-mind-institute-problematic-internet-use\\'

train = pd.read_csv(PATH + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(PATH + 'test.csv', index_col='id', low_memory=True)
data_dict = pd.read_csv(PATH + 'data_dictionary.csv', low_memory=True)
sample = pd.read_csv(PATH + 'sample_submission.csv')

train_ts = load_time_series(PATH + 'series_train.parquet')
test_ts = load_time_series(PATH + 'series_test.parquet')
time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")




train.info(memory_usage='deep')



pciat_min_max = train.groupby('sii')['PCIAT-PCIAT_Total'].agg(['min', 'max'])
col_name = {'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'}
pciat_min_max = pciat_min_max.rename(columns=col_name)
pciat_min_max



data_dict[data_dict['Field'] == 'PCIAT-PCIAT_Total']['Value Labels'].iloc[0]




def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')