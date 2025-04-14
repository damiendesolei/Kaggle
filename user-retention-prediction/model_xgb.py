# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:08:21 2025

@author: zrj-desktop
"""

import warnings
warnings.simplefilter('ignore')
import os
import gc
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
#from gensim.models import Word2Vec
import lightgbm as lgb



PATH = r'G:\\kaggle\user-retention-prediction\\'



# load the 
train = pd.read_csv(PATH+ 'train.csv')

train['DateTime'] = pd.to_datetime(train['Timestamp'], unit='s')

## timezone correction +8h 
train['DateTime'] = train['DateTime'] + pd.Timedelta(hours=8)#'2018-09-21 00:00:00' to '2018-10-20 23:59:59' 

# create month-day combination
train['MonthDay'] = train['DateTime'].dt.month * 100 + train['DateTime'].dt.day

train.tail()



# training set
train_df = train[(train['DateTime'] >= '2018-09-21 00:00:00')&\
                 (train['DateTime'] <= '2018-10-13 23:59:59')].reset_index(drop=True)
    
# active on the last day of training set
active_ids = train_df[(train_df['DateTime'] >= '2018-10-13 00:00:00')&\
                     (train_df['DateTime'] <= '2018-10-13 23:59:59')]['ID'].unique().tolist()
# validation set
valid_df = train[(train['DateTime'] >= '2018-10-14 00:00:00')&\
              (train['DateTime'] <= '2018-10-20 23:59:59')].reset_index(drop=True)
    
# active is there is any action on 2018-10-13
active_df = pd.DataFrame({'ID': active_ids})
mapping = valid_df.groupby('ID')['MonthDay'].nunique().to_dict()
active_df['label'] = active_df['ID'].map(mapping)
active_df['label'].fillna(0, inplace=True)
active_df['label'] = active_df['label'].astype(int)

active_df['label'].value_counts(normalize=True, dropna=False)


### Features ###
def action_feature(train_df, valid_df):
    
    df = valid_df.copy()
    
    # Number of actions per ID
    mapping = train_df.groupby(['ID'])['ID'].count().to_dict()
    df['total_actions'] = df['ID'].map(mapping)
