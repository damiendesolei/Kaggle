# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:25:58 2024

@author: zrj-desktop
"""

import pandas as pd
import numpy as np



X = pd.read_csv(r"G:\kaggle\Functions\Target_Encoding\data\movielens10.csv")
train = X.sample(frac=0.75)
test = X.drop(train.index)

target_encoding_fetas = ['Zipcode']
target = ['Rating']

cumsum = train.groupby(target_encoding_fetas)[target].cumsum() - train[target]
cumcnt = train.groupby(target_encoding_fetas).cumcount()
train['Zipcode_mean_target'] = cumsum / cumcnt
