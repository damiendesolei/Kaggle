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
import joblib 

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
sub = data['SampleSubmissionStage2']
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
features_remove_1 = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 
                     'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 
                     'Pred', 'Score_Diff', 'Score_Diff_Norm', 'WLoc', 'IDTeams_', 'Season', 'season_type'] + c_score_col
features = [c for c in games.columns if c not in features_remove_1]
features_remove_2 = [col for col in games.columns if games[col].isna().mean()>0.5]

FEATURES = [item for item in features if item not in features_remove_1+features_remove_2]

# Assign games to train
train = games[FEATURES+['Pred']]




OUT_PATH = r'G:\\kaggle\march-machine-learning-mania-2025\models\\'


### Xgboost ###
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)
print(xgb.get_config())
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor


# Define the parameter space
import optuna

def objective(trial):
    
    #n_estimators = trial.suggest_int('n_estimators', 2000, 4000, step=200)
    n_estimators = 5_000
    param = {
        'objective': 'reg:logistic',  
        'eval_metric': 'mae', 
        'booster': 'gbtree',
        #'tree_method': 'gpu_hist',
        'device_type': 'cpu',  
        #'gpu_use_dp': True,

        #'n_estimators': 20_000,
        #'n_estimators': trial.suggest_int('n_estimators', 800, 2000, step=200),
        'max_depth': trial.suggest_int('max_depth', 2, 64, step=1),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
        'min_child_weight': trial.suggest_int('min_child_weight', 16, 256, step=2), 

        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        #'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1, log=True),

        'seed': 2025,
        'verbosity': 0,
        "disable_default_eval_metric": 1,  # Disable default eval metric logs
    }

    # Time series cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    scores = []
    
    for i, (train_index, test_index) in enumerate(skf.split(train, train["Pred"])):       
        
        x_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, "Pred"]
        x_valid = train.loc[test_index, FEATURES].copy()
        y_valid = train.loc[test_index, "Pred"]
        #x_test = test[FEATURES].copy()
               
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)

        # Train XGBoost model
        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, 'eval')],
            early_stopping_rounds=25,
            verbose_eval=0
        )
        
        # Predict on validation set
        y_pred = model.predict(dvalid)
    
        mae = mean_absolute_error(y_valid, y_pred)  # MAE for regression
        scores.append(mae)  
    
    mean_mae = np.mean(scores)
    
    return mean_mae


# Run Optuna study
STUDY_XGB = True
N_HOUR = 2
CORES = 6

if STUDY_XGB:
    print("Start running hyper parameter tuning..")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=3600*N_HOUR, n_jobs=CORES)  # 3600*n hour
    
    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best mae:", study.best_value)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Format the file name with the best score
    file_name = f"Xgboost_params_mae_{best_score:.7f}.csv"
    
    # Save the best parameters to a CSV file
    df_param = pd.DataFrame([best_params])  # Convert to DataFrame
    df_param.to_csv(OUT_PATH+file_name, index=False)  # Save to CSV
    
    print(f"Best parameters saved to {file_name}")


    


### Fit Xgboost ###
model_name = f'Xgb_{len(FEATURES)}_features'
# Load the training data
X = train[FEATURES]
y = train['Pred']
test = sub

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=False)

# Initialize variables for OOF predictions, test predictions, and feature importances
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))  # Ensure 'test' DataFrame is loaded
feature_importances = []
models = []
valid_mae = []

if not STUDY_XGB:
# Convert best_params to XGBoost regressor parameters
    xgb_params = {
        'n_estimators': 3000,
        'max_depth': 32,
        'learning_rate': 0.0378768473063554,
        'min_child_weight': 8,
        'colsample_bytree': 0.6,
        'subsample': 1.0,
        'reg_alpha': 0.00249275021430861,
        'reg_lambda': 0.09120328682620336,
    
        'objective': 'reg:logistic',  
        'eval_metric': 'mae', 
        'booster': 'gbtree',
        #'device_type': 'cuda',  
        #'gpu_use_dp': True,
        'seed': 2025
    
        #'use_label_encoder': False
    }
#xgb_params = best_params
# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}")
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    # Initialize and train the model
    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=25)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_valid_fold, y_valid_fold)],  
              verbose=500)

    # Generate validation predictions
    y_pred_valid = model.predict_proba(X_valid_fold)[:, 1]  # Get probability of positive class
    fold_mae = mean_absolute_error(y_valid_fold, y_pred_valid)
    valid_mae.append(fold_mae)
    oof_predictions[valid_idx] = y_pred_valid

    # Generate test predictions and accumulate
    test_pred = model.predict_proba(test[features])[:, 1]
    test_predictions += test_pred / skf.n_splits

    # Save model and feature importance
    joblib.dump(model, f"{OUT_PATH}{model_name}_fold{fold+1}_mae_{fold_mae:.7f}.model")
    models.append(model)
    
    # Collect feature importances
    fold_importance = pd.DataFrame({
        'Feature': model.get_booster().feature_names,
        'Importance': model.feature_importances_,
        'Fold': fold + 1
    })
    feature_importances.append(fold_importance)

    print(f"Fold {fold + 1} MAE: {fold_mae:.5f}")

# Calculate overall metrics
overall_mae= mean_absolute_error(y, oof_predictions)
print(f"Average Validation MAE: {np.mean(valid_mae):.7f}")
print(f"Overall OOF MAE: {overall_mae:.7f}")

# Save OOF predictions and true values
oof_df = X.copy()
oof_df['Result'] = y
oof_df['prediction'] = oof_predictions
oof_df.to_csv(f"{OUT_PATH}{model_name}_oof_predictions_mae_{overall_mae:.7f}.csv", index=False)

# Aggregate and save feature importances
feature_importances_df = pd.concat(feature_importances)
average_importance = feature_importances_df.groupby('Feature')['Importance'].mean().reset_index()
average_importance = average_importance.sort_values('Importance', ascending=False)
average_importance.to_csv(f"{OUT_PATH}{model_name}_average_feature_importance.csv", index=False)

# Plot average feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=average_importance.head(25))
plt.title('Xgb Top Features (Average Importance)')
plt.tight_layout()
plt.show()



#https://www.kaggle.com/code/takuji/march-mania-2025-tutorial-japanese
import statsmodels.api as sm


#### Load data ####

DATA_PATH = r'G:\\kaggle\march-machine-learning-mania-2025\\'

tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)

regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
], ignore_index=True)


#### Feature Engineering ####
def prepare_data(df):
    # Prepare a winner and loser swap.
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    
      
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    # Combine the original data and the swapped data
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


#First, the feature quantities of game stats for the season
boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff']

funcs = "mean"

season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()

season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"


# Attach the features you created to the right of the March Madness data.
# Unfortunately, the March Madness stats information has been deleted. It is of no use because you cannot obtain information for that year.
tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]

tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

# win ratio of last 14 days
last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

# Attach this to the right of the March Madness data
tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

# Creating a feature called team quality
regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])

def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    #quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality

formula = 'win~-1+T1_TeamID+T2_TeamID'
glm = sm.GLM.from_formula(formula=formula, 
                          data=regular_season_effects.loc[regular_season_effects.Season==2010,:], 
                          family=sm.families.Binomial(link=sm.families.links.Logit())).fit()  # Default is logit

quality = pd.DataFrame(glm.params).reset_index()

glm_quality = pd.concat([team_quality(2010),
                         team_quality(2011),
                         team_quality(2012),
                         team_quality(2013),
                         team_quality(2014),
                         team_quality(2015),
                         team_quality(2016),
                         team_quality(2017),
                         team_quality(2018),
                         team_quality(2019),
                         ##team_quality(2020),
                         team_quality(2021),
                         team_quality(2022),
                         team_quality(2023),
                         team_quality(2024)
                         ]).reset_index(drop=True)

# Attach quality feature to tourney data
glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']

tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# Seed as a feature
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

# Attach Seed feature to tourney data
seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']

tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

# できあがった特徴量の数をチェック
features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    list(last14days_stats_T1.columns[2:999]) + \
    list(last14days_stats_T2.columns[2:999]) + \
    ["Seed_diff"] + ["T1_quality","T2_quality"]

print("No of features is:", len(features))

# The resulting features look like this:
display(tourney_data[features].head(3))
print("\n features shape:", tourney_data[features].shape)




#### Xgboost ####

import xgboost as xgb
print("Using XGBoost version",xgb.__version__)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# predict the point difference.
#y = tourney_data['T1_Score'] - tourney_data['T2_Score']
#X = tourney_data[features].values
tourney_data['Pred'] = tourney_data['T1_Score'] - tourney_data['T2_Score']
train = tourney_data[features+['Pred']]


# Define the parameter space
import optuna
OUT_PATH = r'G:\\kaggle\march-machine-learning-mania-2025\models\\'

def objective(trial):
    
    n_estimators = trial.suggest_int('n_estimators', 1500, 3500, step=100)
    #n_estimators = 3_000
    param = {
        'objective': 'reg:squarederror',  
        'eval_metric': 'mae', 
        'booster': 'gbtree',
        'device_type': 'cpu',  
        #'gpu_use_dp': True,

        #'n_estimators': 20_000,
        #'n_estimators': trial.suggest_int('n_estimators', 800, 2000, step=200),
        'max_depth': trial.suggest_int('max_depth', 1, 16, step=1),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
        'min_child_weight': trial.suggest_int('min_child_weight', 12, 128, step=2), 

        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        #'bagging_freq': trial.suggest_int('bagging_freq', 2, 12),  
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1, log=True),

        'seed': 2025,
        'verbosity': 0,
        "disable_default_eval_metric": 1,  # Disable default eval metric logs
    }

    # Time series cross-validation
    skf = KFold(n_splits=5, shuffle=False)
    scores = []
    
    for i, (train_index, test_index) in enumerate(skf.split(train, train["Pred"])):       
        
        x_train = train.loc[train_index, features].copy()
        y_train = train.loc[train_index, "Pred"]
        x_valid = train.loc[test_index, features].copy()
        y_valid = train.loc[test_index, "Pred"]
        #x_test = test[FEATURES].copy()
               
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)

        # Train XGBoost model
        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, 'eval')],
            early_stopping_rounds=25,
            verbose_eval=0
        )
        
        # Predict on validation set
        y_pred = model.predict(dvalid)
    
        mae = mean_absolute_error(y_valid, y_pred)  # MAE for regression
        scores.append(mae)  
    
    mean_mae = np.mean(scores)
    
    return mean_mae


# Run Optuna study
STUDY_XGB_1 = True
N_HOUR = 9
CORES = 6

if STUDY_XGB_1:
    print("Start running hyper parameter tuning..")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=3600*N_HOUR, n_jobs=CORES)  # 3600*n hour
    
    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best mae:", study.best_value)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Format the file name with the best score
    file_name = f"Xgboost2_params_mae_{best_score:.7f}.csv"
    
    # Save the best parameters to a CSV file
    df_param = pd.DataFrame([best_params])  # Convert to DataFrame
    df_param.to_csv(OUT_PATH+file_name, index=False)  # Save to CSV
    
    print(f"Best parameters saved to {file_name}")




### Fit Xgboost2 with cauchy loss ###
y = tourney_data['T1_Score'] - tourney_data['T2_Score']
X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)

# cauchy loss
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess

if STUDY_XGB_1:
    param = best_params


param = {} 
#param['objective'] = 'reg:linear'
param['eval_metric'] =  'mae'
param['booster'] = 'gbtree'
#param['n_estimators'] = 1500,
param['eta'] = 0.060920150610591785
param['colsample_bytree'] = 0.9
param['subsample'] = 0.6
#param['num_parallel_tree'] = 3 #recommend 10
param['min_child_weight'] = 16
#param['gamma'] = 10
param['max_depth'] =  1
param['reg_alpha'] =  0.0022760750736765946
param['reg_lambda'] =  0.0048449207058357865
#param['silent'] = 1


# Fit the model on cv = 5
xgb_cv = []
repeat_cv = 3 # recommend 10

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3000,
          folds = KFold(n_splits = 5, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 500
        )
    )

iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]

# Save the predictions for the validation data.
oof_preds = []
for i in range(repeat_cv):
    preds = y.copy()
    kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
    for train_index, val_index in kfold.split(X,y):
        dtrain_i = xgb.DMatrix(X[train_index], label = y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label = y[val_index])  
        model = xgb.train(
              params = param,
              dtrain = dtrain_i,
              num_boost_round = iteration_counts[i],
              verbose_eval = 50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds,-30,30))
    
    
    
    
    
#### Spline Model ####
from scipy.interpolate import UnivariateSpline

spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))

# スプラインを可視化するとこんな感じ
plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int').mean().reset_index()

plt.figure(figsize=[5.3,3.0])
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)





#### Make Predictions ####
sub = pd.read_csv(DATA_PATH + "SampleSubmissionStage2.csv")
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))

# Features
sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]

# Form Xgboost dataset
Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)

# retraining the model using all training data 
sub_models = []
for i in range(repeat_cv):
    sub_models.append(
        xgb.train(
          params = param,
          dtrain = dtrain,
          num_boost_round = int(iteration_counts[i] * 1.05),
          verbose_eval = 50
        )
    )

# Ensemble Xgboost2 and Spline
sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.025,0.975))
    
sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
sub[['ID','Pred']].to_csv(OUT_PATH+"Xgboost2_Spline_submission.csv", index = None)

display(sub[['ID','Pred']].head(3))