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


# each team's # of wins 
num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})

# number of wins in the last 3, 5, 10 years


# each team's # of loss
num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})


# number of losses in the last 3, 5, 10 years



# how much points they scored more in average
gap_win = df_season_results.groupby(['Season', 'WTeamID']).mean('ScoreGap').reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})


# how much points they scored less in average
gap_loss = df_season_results.groupby(['Season', 'LTeamID']).mean('ScoreGap').reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})


df_features_season_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)

df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')


df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['GapAvg'] = ((df_features_season['NumWins'] * df_features_season['GapWins'] - df_features_season['NumLosses'] * df_features_season['GapLosses'])
    / (df_features_season['NumWins'] + df_features_season['NumLosses']))



# tourney_results
df_tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "WNCAATourneyCompactResults.csv"),
    pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv"),
], ignore_index=True)
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


df = df_tourney_results.copy()
df = df[df['Season'] >= 2016].reset_index(drop=True)
df.head()


df_seeds.head()
df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']
              ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})

df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']
              ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})


def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))

df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)





df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']
              ).rename(columns={'NumWins': 'NumWinsW',
                                'NumLosses': 'NumLossesW',
                                'GapWins': 'GapWinsW',
                                'GapLosses': 'GapLossesW',
                                'WinRatio': 'WinRatioW',
                                'GapAvg': 'GapAvgW'}
                                ).drop(columns='TeamID', axis=1)
                                
                                
df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']
              ).rename(columns={'NumWins': 'NumWinsL',
                                'NumLosses': 'NumLossesL',
                                'GapWins': 'GapWinsL',
                                'GapLosses': 'GapLossesL',
                                'WinRatio': 'WinRatioL',
                                'GapAvg': 'GapAvgL'}
                                ).drop(columns='TeamID', axis=1)   
                                
                                
                                
                                
                                
df_test = pd.read_csv(DATA_PATH  + "/sample_submission.csv")


#https://www.kaggle.com/code/samdaltonjr/preliminary-preds-into-bracket/notebook

preds = pd.read_csv(DATA_PATH + 'submission_rustyb.csv')
round_slots = pd.read_csv(DATA_PATH + 'MNCAATourneySeedRoundSlots.csv')
seeds = pd.read_csv(DATA_PATH + '2024_tourney_seeds.csv')

preds['Year'], preds['Team1ID'], preds['Team2ID'] = zip(*preds['ID'].apply(lambda x: x.split('_')).tolist())