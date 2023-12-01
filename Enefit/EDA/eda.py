# -*- coding: utf-8 -*-


######Nov-11-2023######
####Shanghai-China#####

import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import holidays

from pathlib import Path

data_path = Path('C:/Users/damie/Downloads/predict-energy-behavior-of-prosumers')

train = pd.read_csv(data_path / 'train.csv', parse_dates=['datetime'])
train.info(memory_usage='deep')

train.head()

for col in train.columns:
    print(str(col) + ' has unique values of:')
    print(set(train[col].unique()))