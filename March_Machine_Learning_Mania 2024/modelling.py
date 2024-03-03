# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:02:17 2024

@author: zrj-desktop
"""

import os
import re

import pandas as pd
#import polars as pl

# Show all properties on display and set style
pd.set_option('display.max_columns', None)
DATA_PATH = 'G:/kaggle/March_Machine_Learning_Mania_2024/march-machine-learning-mania-2024/'

for filename in sorted(os.listdir(DATA_PATH)):
    print(filename)
    
    
    
df_seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)
    
df_seeds.head()




df_season_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv"),
], ignore_index=True)

df_season_results


# scoregap between W & L
df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']
df_season_results.head()

# wining margin
df_season_results['WinMargin'] = df_season_results['ScoreGap'] / df_season_results['LScore']
df_season_results.head()
