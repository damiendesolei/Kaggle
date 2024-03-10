# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:02:17 2024

@author: zrj-desktop
"""

import os
import re

import pandas as pd
import numpy as np
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


# Function to simulate a single matchup
def simulate_matchup(probability):
    return np.random.rand() < probability  # Returns True if Team1 wins, False if Team2 wins

#sample_submission as template for the output
sample_submission = pd.read_csv(DATA_PATH  + "/sample_submission.csv")

submission = sample_submission.copy()
submission.rename(columns={'Team':'Winner'},inplace=True)

# read in new seeds
tourney_slots = pd.read_csv(DATA_PATH + '/MNCAATourneySlots.csv')
#filter out tourney_slots to only include 2023
tourney_slots = tourney_slots[tourney_slots['Season'] == 2023]

tourney_slots_M = tourney_slots.copy()
tourney_slots_M['Tournament'] = 'M'

tourney_slots_W = tourney_slots.copy()
tourney_slots_W['Tournament'] = 'W'

tourney_slots = pd.concat([tourney_slots_M, tourney_slots_W])


def create_round(slot):
    if slot[0] != 'R':
        return 0
    elif slot[0] == 'R':
        return int(slot[1])
    
tourney_slots['Round'] = tourney_slots['Slot'].apply(create_round)
tourney_slots = tourney_slots[tourney_slots['Round'] != 0]

preds_df = preds_w_seeds[['ID','new_ID','Year','HigherSeedID','LowerSeedID','Tournament','Pred']]

preds_df


#merge preds_df with initial_matchups to get the pred for round 1 matchup

tourney_2023 = tourney_slots.copy()

#find round 1 matchups
initial_matchups = tourney_2023[tourney_2023['Round'] == 1]

results = initial_matchups.merge(preds_df, left_on=['StrongSeed','WeakSeed','Season'], right_on=['HigherSeedID','LowerSeedID','Year'], how='left')

results.rename(columns={'Tournament_x':'Tournament'}, inplace=True)
#drop Touranment_y column  
results.drop('Tournament_y', axis=1, inplace=True)


#simulate round 1 matchups, if simulate_matchup results in True, winner is the higher seed, if False, winner is the lower seed
results['outcome'] = results['Pred'].apply(simulate_matchup)

results['Team'] = results.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

#drop duplicates from results
results.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

results

round_1_results = results[['Tournament','Slot','Team','ID']]

round_1_results


#take Team from round_1_results and merge with tourney_2023 to get the next round matchups
round_2 = tourney_2023[tourney_2023['Round'] == 2]

round_2 = round_2.merge(round_1_results, left_on=['StrongSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_2.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_2.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_2.drop_duplicates(subset=['Slot','Tournament'], inplace=True)


round_2 = round_2.merge(round_1_results, left_on=['WeakSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_2.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_2.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_2.drop_duplicates(subset=['Slot','Tournament'], inplace=True)


#if team_x is the lower seed, set StrongSeed to Team_x and WeakSeed to Team_y, else set StrongSeed to Team_y and WeakSeed to Team_x
round_2['StrongSeed'] = round_2.apply(lambda x: x['Team_x'] if x['Team_x'] < x['Team_y'] else x['Team_y'], axis=1)
round_2['WeakSeed'] = round_2.apply(lambda x: x['Team_x'] if x['Team_x'] > x['Team_y'] else x['Team_y'], axis=1)

#drop Team_x and Team_y
round_2.drop(['Team_x', 'Team_y'], axis=1, inplace=True)


#merge round_2 with preds_df to get the pred for round 2 matchups
round_2 = round_2.merge(preds_df, left_on=['StrongSeed','WeakSeed','Tournament'], right_on=['HigherSeedID','LowerSeedID','Tournament'], how='left')

round_2['outcome'] = round_2['Pred'].apply(simulate_matchup)

round_2['Team'] = round_2.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

round_2

round_2_results = round_2[['Tournament','Slot','Team','ID']]

round_2_results


round_3 = tourney_2023[tourney_2023['Round'] == 3]

round_3 = round_3.merge(round_2_results, left_on=['StrongSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_3.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_3.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_3.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

round_3 = round_3.merge(round_2_results, left_on=['WeakSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_3.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_3.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_3.drop_duplicates(subset=['Slot','Tournament'], inplace=True)


round_3['StrongSeed'] = round_3.apply(lambda x: x['Team_x'] if x['Team_x'] < x['Team_y'] else x['Team_y'], axis=1)
round_3['WeakSeed'] = round_3.apply(lambda x: x['Team_x'] if x['Team_x'] > x['Team_y'] else x['Team_y'], axis=1)

#drop Team_x and Team_y
round_3.drop(['Team_x', 'Team_y'], axis=1, inplace=True)


round_3 = round_3.merge(preds_df, left_on=['StrongSeed','WeakSeed','Tournament'], right_on=['HigherSeedID','LowerSeedID','Tournament'], how='left')

round_3['outcome'] = round_3['Pred'].apply(simulate_matchup)

round_3['Team'] = round_3.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

round_3

round_3_results = round_3[['Tournament','Slot','Team','ID']]
round_3_results


round_4 = tourney_2023[tourney_2023['Round'] == 4]

round_4 = round_4.merge(round_3_results, left_on=['StrongSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_4.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_4.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_4.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

round_4 = round_4.merge(round_3_results, left_on=['WeakSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_4.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_4.drop(['Slot_y'], axis=1, inplace=True)
    
#drop duplicates
round_4.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

round_4['StrongSeed'] = round_4.apply(lambda x: x['Team_x'] if x['Team_x'] < x['Team_y'] else x['Team_y'], axis=1)
round_4['WeakSeed'] = round_4.apply(lambda x: x['Team_x'] if x['Team_x'] > x['Team_y'] else x['Team_y'], axis=1)

#drop Team_x and Team_y
round_4.drop(['Team_x', 'Team_y'], axis=1, inplace=True)

round_4 = round_4.merge(preds_df, left_on=['StrongSeed','WeakSeed','Tournament'], right_on=['HigherSeedID','LowerSeedID','Tournament'], how='left')

round_4['outcome'] = round_4['Pred'].apply(simulate_matchup)

round_4['Team'] = round_4.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

round_4

round_4_results = round_4[['Tournament','Slot','Team','ID']]

round_4_results


round_5 = tourney_2023[tourney_2023['Round'] == 5]

round_5 = round_5.merge(round_4_results, left_on=['StrongSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_5.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_5.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_5.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

round_5 = round_5.merge(round_4_results, left_on=['WeakSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_5.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_5.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_5.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

def compare_seeds(seed1, seed2):
    seed_num1 = extract_seed_number(seed1)
    seed_num2 = extract_seed_number(seed2)

    if seed_num1 == seed_num2:
        # If the seed numbers are equal, compare the letters
        return seed1 if seed1 < seed2 else seed2
    else:
        # Otherwise, compare the seed numbers
        return seed1 if seed_num1 < seed_num2 else seed2

# Apply comparison logic to determine StrongSeed and WeakSeed
round_5['StrongSeed'] = round_5.apply(lambda x: compare_seeds(x['Team_x'], x['Team_y']), axis=1)
round_5['WeakSeed'] = round_5.apply(lambda x: x['Team_x'] if x['Team_x'] != x['StrongSeed'] else x['Team_y'], axis=1)

#drop Team_x and Team_y
round_5.drop(['Team_x', 'Team_y'], axis=1, inplace=True)
round_5


round_5 = round_5.merge(preds_df, left_on=['StrongSeed','WeakSeed','Tournament'], right_on=['HigherSeedID','LowerSeedID','Tournament'], how='left')

round_5['outcome'] = round_5['Pred'].apply(simulate_matchup)

round_5['Team'] = round_5.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

round_5

round_5_results = round_5[['Tournament','Slot','Team','ID']]

round_5_results


round_6 = tourney_2023[tourney_2023['Round'] == 6]

round_6 = round_6.merge(round_5_results, left_on=['StrongSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_6.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)
#drop columns from merge
round_6.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_6.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

round_6 = round_6.merge(round_5_results, left_on=['WeakSeed','Tournament'], right_on=['Slot','Tournament'], how='left')
#rename original columns
round_6.rename(columns={'Slot_x':'Slot', 'Tournament_x':'Tournament'}, inplace=True)    
#drop columns from merge
round_6.drop(['Slot_y'], axis=1, inplace=True)

#drop duplicates
round_6.drop_duplicates(subset=['Slot','Tournament'], inplace=True)

# Apply comparison logic to determine StrongSeed and WeakSeed
round_6['StrongSeed'] = round_6.apply(lambda x: compare_seeds(x['Team_x'], x['Team_y']), axis=1)
round_6['WeakSeed'] = round_6.apply(lambda x: x['Team_x'] if x['Team_x'] != x['StrongSeed'] else x['Team_y'], axis=1)

round_6.drop(['Team_x', 'Team_y'], axis=1, inplace=True)

round_6 = round_6.merge(preds_df, left_on=['StrongSeed','WeakSeed','Tournament'], right_on=['HigherSeedID','LowerSeedID','Tournament'], how='left')

round_6['outcome'] = round_6['Pred'].apply(simulate_matchup)

round_6['Team'] = round_6.apply(lambda x: x['HigherSeedID'] if x['outcome'] == True else x['LowerSeedID'], axis=1)

round_6

round_6_results = round_6[['Tournament','Slot','Team','ID']]


all_rounds = pd.concat([round_1_results, round_2_results, round_3_results, round_4_results, round_5_results, round_6_results])

all_rounds

round_6_results


#sample_submission as template for the output

df = sample_submission.copy()

#drop Team column from sample_submission
df.drop('Team', axis=1, inplace=True)

#merge sample_submission with all_rounds on Tournament and Slot
submission = df.merge(all_rounds, left_on=['Tournament', 'Slot'], right_on=['Tournament', 'Slot'], how='left')

#drop ID from submission
submission.drop('ID', axis=1, inplace=True)

submission

#return all null rows from submission
submission[submission['Team'].isnull()]