# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#https://www.kaggle.com/code/sadettinamilverdil/ncaa-basketball-predictions-with-xgboost



import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from xgboost import XGBRegressor

#import warnings
#warnings.filterwarnings("ignore")



### Load the data ###
PATH = r'G:\kaggle\march-machine-learning-mania-2025'  # Ensure consistent use of raw string
file_paths = glob.glob(PATH + '/*.csv')  # Look for all CSV files in the directory
data = {p.split('\\')[-1].split('.')[0]: pd.read_csv(p, encoding='utf-8') for p in file_paths}

## Teams
teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])

## Season Results
season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])

## Tourney Results
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])

## Slots
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])

## Seed
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}

## city
gcities = pd.concat([data['MGameCities'], data['WGameCities']])

## Season
seasons = pd.concat([data['MSeasons'], data['WSeasons']])



### Feature engineering ###

# Flag Season
season_cresults['season_type'] = 'S'
season_dresults['season_type'] = 'S'
# Flag Tourney
tourney_cresults['season_type'] = 'T'
tourney_dresults['season_type'] = 'T'

games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games = games[games.Season>=2019]  # only include recent data

games.reset_index(drop=True, inplace=True)
print('games after filtering for Season>= 2019\n has shape',games.shape)

# integer encode WLoc
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

# create multiple keys 
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# map seeds to teams
games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

# Assign 1 to wining team, 0 otherwise
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
# calculate the winning margin
games['Score_Diff'] = games['WScore'] - games['LScore']
games['Score_Diff_Norm'] = games.apply(lambda r: r['Score_Diff'] * -1 if r['Pred'] == 0. else r['Score_Diff'], axis=1)
# calculate seed diff
games['Seed_Diff'] = games['Team1Seed'] - games['Team2Seed'] 


# Summary statistics
c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
               'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 
               'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
game_stats = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
game_stats.columns = ['_'.join(c) for c in game_stats.columns]


# Load the submission file
sub = data['SampleSubmissionStage1']
sub['WLoc'] = 3 # Neutral Court
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)

sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])

sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# Map seed to the year_team
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['Seed_Diff'] = sub['Team1Seed'] - sub['Team2Seed'] 


# Attach game statistics
games = games.merge(game_stats, how='left', left_on='IDTeams', right_on='IDTeams_')
sub = sub.merge(game_stats, how='left', left_on='IDTeams', right_on='IDTeams_')





### Features ###
features = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc', 'IDTeams_'] + c_score_col]
features_remove = [col for col in games.columns if games[col].isna().mean()>0.5]

