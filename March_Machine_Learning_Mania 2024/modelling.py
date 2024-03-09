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
preds['Team1ID'] = preds['Team1ID'].astype(int)
preds['Team2ID'] = preds['Team2ID'].astype(int)
preds['Year'] = preds['Year'].astype(int)

seeds['Year'] = 2023

# Merge seeds with preds for both teams
preds = preds.merge(seeds, left_on=['Year', 'Team1ID'], right_on=['Year', 'TeamID'], how='left')
preds.rename(columns={'Seed': 'Team1Seed'}, inplace=True)
preds.drop('TeamID', axis=1, inplace=True)

preds = preds.merge(seeds, left_on=['Year', 'Team2ID'], right_on=['Year', 'TeamID'], how='left')
preds.rename(columns={'Seed': 'Team2Seed', 'Tournament_x':'Tournament'}, inplace=True)
preds.drop(['TeamID', 'Tournament_y'], axis=1, inplace=True)

#drop rows where pred has null values
preds = preds.dropna()

#This cell is where the transformation of your original prediction file takes place, it flips the {Year}_{Team1ID}_{Team2ID} format into {Year}_{HigherSeed}_{LowerSeed}
#An Additional Column new_ID will be created to comtain the original Team IDs

preds_w_seeds = preds.copy()

#sort preds_w_seeds by Team1Seed
preds_w_seeds = preds_w_seeds.sort_values(by='Team1Seed')

#Flip preds to where pred is based on the higher seed and not lower seed, so pred must be transformed accordingly but new_ID would be formated as 2023_X01_X16, etc. only where pred is not already in the correct format

def extract_seed_number(seed):
    return int(seed[1:])


# Assuming preds_w_seeds is your DataFrame
# Define a function to compare seeds
def compare_seeds(seed1, seed2):
    seed1_num = extract_seed_number(seed1)
    seed2_num = extract_seed_number(seed2)
    seed1_prefix = seed1[0]
    seed2_prefix = seed2[0]

    if seed1_num < seed2_num:
        return -1
    elif seed1_num > seed2_num:
        return 1
    else:  # If seed numbers are equal, compare prefixes
        if seed1_prefix < seed2_prefix:
            return -1
        elif seed1_prefix > seed2_prefix:
            return 1
        else:
            return 0

# Assuming preds_w_seeds is your DataFrame
# First, identify higher seed and lower seed for each matchup
def determine_seeds(row):
    seed1 = row['Team1Seed']
    seed2 = row['Team2Seed']

    comparison = compare_seeds(seed1, seed2)
    if comparison < 0:
        row['HigherSeed'] = row['Team1ID']
        row['HigherSeedID'] = seed1
        row['LowerSeed'] = row['Team2ID']
        row['LowerSeedID'] = seed2
    elif comparison > 0:
        row['HigherSeed'] = row['Team2ID']
        row['HigherSeedID'] = seed2
        row['LowerSeed'] = row['Team1ID']
        row['LowerSeedID'] = seed1
    else:  # If seeds are equal
        if row['Team1ID'] < row['Team2ID']:
            row['HigherSeed'] = row['Team1ID']
            row['HigherSeedID'] = seed1
            row['LowerSeed'] = row['Team2ID']
            row['LowerSeedID'] = seed2
        else:
            row['HigherSeed'] = row['Team2ID']
            row['HigherSeedID'] = seed2
            row['LowerSeed'] = row['Team1ID']
            row['LowerSeedID'] = seed1

    return row

preds_w_seeds = preds_w_seeds.apply(determine_seeds, axis=1)

# Then, rearrange the data to create new ID and update Pred column accordingly
preds_w_seeds['ID'] = preds_w_seeds.apply(lambda x: f"{x['Year']}_{x['HigherSeed']}_{x['LowerSeed']}", axis=1)
preds_w_seeds['Pred'] = preds_w_seeds.apply(lambda x: 1 - x['Pred'] if x['Team1ID'] != x['HigherSeed'] else x['Pred'], axis=1)  # Flip Pred only if teams are rearranged
preds_w_seeds = preds_w_seeds[['ID', 'Pred', 'Year', 'HigherSeedID', 'LowerSeedID', 'Tournament', 'HigherSeed', 'LowerSeed']]

preds_w_seeds['new_ID'] = preds_w_seeds['Tournament'] + '_' + preds_w_seeds['Year'].astype(str) + '_' + preds_w_seeds['HigherSeedID'] + '_' + preds_w_seeds['LowerSeedID']
