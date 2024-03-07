# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:20:31 2024

@author: zrj-desktop

"""
#https://aistudio.baidu.com/projectdetail/3795293

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


X = pd.read_csv(r"G:\kaggle\Functions\Target_Encoding\data\movielens10.csv")
train = X.sample(frac=0.5)
test = X.drop(train.index)

target_encoding_fetas = ['Zipcode']



def gen_target_encoding_feats(train, test, encode_cols, target_col, n_fold=10):
    #https://aistudio.baidu.com/projectdetail/3795293#
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values
    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]
    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict)
    return train, test


train, test = gen_target_encoding_feats(train, test, target_encoding_fetas, target_col='Rating', n_fold=2)
