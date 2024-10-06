# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:24:10 2024

@author: damie
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import *
from sklearn.metrics import *

from tqdm import tqdm



#PATH = 'G:\kaggle\Classification_with_an_Academic_Success_Dataset\\'
PATH = r'C:\Users\damie\Downloads\child-mind-institute-problematic-internet-use\\'

train = pd.read_csv(PATH + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(PATH + 'test.csv', index_col='id', low_memory=True)
data_dict = pd.read_csv(PATH + 'data_dictionary.csv', low_memory=True)

train.info(memory_usage='deep')



pciat_min_max = train.groupby('sii')['PCIAT-PCIAT_Total'].agg(['min', 'max'])
col_name = {'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'}
pciat_min_max = pciat_min_max.rename(columns=col_name)
pciat_min_max



data_dict[data_dict['Field'] == 'PCIAT-PCIAT_Total']['Value Labels'].iloc[0]


#https://www.kaggle.com/code/abdmental01/cmi-best-single-model
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')