# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

import xgboost
import lightgbm
import catboost

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import optuna

#import statsmodels.api as sm
#import statsmodels.formula.api as smf


import matplotlib.pyplot as plt
import datetime
from colorama import Fore, Style


#import seaborn as sns




##############################################
######### mean encoding function  ############
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing - sigmoid funttion
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
######### end of encoding function ###########
##############################################




##############################################
######### read in data #######################
PATH = r'G:\\kaggle\Binary_Classification_of_Insurance_Cross_Selling\\'

train = pd.read_csv(PATH + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(PATH + 'test.csv', index_col='id', low_memory=True)
######### finish reading in data #############
##############################################




##############################################
######### glimpse of data ####################

#check dimensions
print(f'train dataFrame size: {train.shape}')
print(f'test dataFrame size: {test.shape}')

#check na
print(f'Number of missing values in train:\n{train.isna().sum()}')
print(f'Number of missing values in test:\n{test.isna().sum()}')

#check categoricals
categorical_columns = train.select_dtypes(include=['object']).columns
unique_counts = train[categorical_columns].nunique()
print(unique_counts)

#feature distribution
train.info()
train.describe().T
train.Gender.value_counts(normalize=True)
train.Vehicle_Damage.value_counts(normalize=True)
train.Vehicle_Age.value_counts(normalize=True)
######### finish glimpse pf data #############
##############################################





##############################################
######### encoding object columns ############
#integer encode the Gender
def int_encode_gender(col):
    if col == 'Male':
        return 0
    elif col == 'Female':
        return 1
    else:
        return 9999
#train.groupby('Gender').count()
train['Gender_IntEncoded'] = train['Gender'].apply(int_encode_gender)
test['Gender_IntEncoded'] = test['Gender'].apply(int_encode_gender)
test[['Gender','Gender_IntEncoded']].drop_duplicates() #show encoding


#integer encode the Vehicle_Age
def int_encode_vehicle_age(col):
    if col == '< 1 Year':
        return 0
    elif col == '1-2 Year':
        return 1
    elif col == '> 2 Years':
        return 2  
    else:
        return 9999
#train.groupby('Gender').count()
train['Vehicle_Age_IntEncoded'] = train['Vehicle_Age'].apply(int_encode_vehicle_age)
test['Vehicle_Age_IntEncoded'] = test['Vehicle_Age'].apply(int_encode_vehicle_age)
test[['Vehicle_Age','Vehicle_Age_IntEncoded']].drop_duplicates() #show encoding


#integer encode the Vehicle_Damage
def int_encode_vehicle_damage(col):
    if col == 'No':
        return 0
    elif col == 'Yes':
        return 1
    else:
        return 9999   
#train.groupby('Gender').count()
train['Vehicle_Damage_IntEncoded'] = train['Vehicle_Damage'].apply(int_encode_vehicle_damage)
test['Vehicle_Damage_IntEncoded'] = test['Vehicle_Damage'].apply(int_encode_vehicle_damage)
test[['Vehicle_Damage','Vehicle_Damage_IntEncoded']].drop_duplicates() #show encoding


#mean encode Policy_Sales_Channel
trn_, tst_ = target_encode(train['Policy_Sales_Channel'], 
                           test['Policy_Sales_Channel'], 
                           target=train.Response, 
                           min_samples_leaf=330000,
                           smoothing=80000,
                           noise_level=0)
train['Policy_Sales_Channel_TargetEncoded'] = trn_
test['Policy_Sales_Channel_TargetEncoded'] = tst_
train[['Policy_Sales_Channel','Policy_Sales_Channel_TargetEncoded']].head(10)


#mean encode Vintage
trn_, tst_ = target_encode(train['Vintage'], 
                           test['Vintage'], 
                           target=train.Response, 
                           min_samples_leaf=330000,
                           smoothing=950000,
                           noise_level=0)
train['Vintage_TargetEncoded'] = trn_
test['Vintage_TargetEncoded'] = tst_
train[['Vintage','Vintage_TargetEncoded']].head(10)
######### finish encoding columns ############
##############################################


# def encoder(df):
#     gender_map = {
#         'Female': 0,
#         'Male': 1
#     }

#     vehicle_age_map = {
#         '< 1 Year': 0,
#         '1-2 Year': 1,
#         '> 2 Years': 2
#     }

#     vehicle_damage_map = {
#         'No': 0,
#         'Yes': 1
#     }

#     df['Gender'] = df['Gender'].map(gender_map).astype(np.int8)
#     df['Driving_License'] = df['Driving_License'].astype(np.int8)
#     df['Previously_Insured'] = df['Previously_Insured'].astype(np.int8)
#     df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map).astype(np.int8)
#     df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vehicle_damage_map).astype(np.int8)
    
#     return df





##############################################
######### drop certain columns ###############
drop_list = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vintage']

def drop_columns(df, drop_list):
    for col in drop_list:
        try:
            df.drop([col], axis=1, inplace=True)
            print(f'{col} is dropped')
        except:
            print(f'{col} does not exist')
    return df

drop_columns(train, drop_list)        
drop_columns(test, drop_list) 
      
initial_features = list(train.columns)
if "Response" in initial_features:
    initial_features.remove("Response")

#set aside holdout sets for fitting ensemble weight
train, holdout = train_test_split(train, test_size=0.2, stratify=train.Response , random_state = 24)
train = train.reset_index(drop=True)
holdout = holdout.reset_index(drop=True)

##############################################
######### finish dropping columns ############






##############################################
######### modelling setup  ###################
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)

def cross_validate(model, label, features=initial_features):
    """Compute out-of-fold and test predictions for a given model.
    
    Out-of-fold and test predictions are stored in the global variables
    oof and test_pred, respectively. 
    """
    start_time = datetime.datetime.now()
    tr_scores = []
    va_scores = []
    oof_preds = np.full_like(train.Response, np.nan, dtype=np.float64)
    #for fold, (idx_tr, idx_va) in enumerate(kf.split(train)):
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.Response)):    
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train.Response[idx_tr]
        y_va = train.Response[idx_va]
        
        model.fit(X_tr, y_tr)
        #y_pred = model.predict(X_va)
        y_pred = model.predict_proba(X_va)
        #y_predicted = np.argmax(y_pred, axis=1) #find the highest probability row array position 
        y_predicted = y_pred[:,1] #find the probability of being 1
        

        va_score = roc_auc_score(y_va, y_predicted)
        #tr_score = roc_auc_score(y_tr, model.predict(X_tr))
        #tr_score = roc_auc_score(y_tr, np.argmax(model.predict_proba(X_tr), axis=1))
        tr_score = roc_auc_score(y_tr, model.predict_proba(X_tr)[:,1])
        print(f"# Fold {fold}: tr_auc={tr_score:.5f}, val_auc={va_score:.5f}")

        va_scores.append(va_score)
        tr_scores.append(tr_score)
        oof_preds[idx_va] = y_predicted #each iteration will fill in 1/5 of the index
        #oof_preds[idx_va] = y_pred
            
    elapsed_time = datetime.datetime.now() - start_time
    print(f"{Fore.RED}# Overall val={np.array(va_scores).mean():.5f} {label}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    print(f"{Fore.RED}# {label} Fitting started from {start_time}")
    oof[label] = oof_preds

    if COMPUTE_HOLDOUT_PRED:
        X_ho = holdout[features]
        y_ho = holdout.Response
        model.fit(X_ho, y_ho)
        y_pred = model.predict_proba(holdout[features])
        #y_predicted = np.argmax(y_pred, axis=1)
        y_predicted = y_pred[:,1]
        #holdout_pred[label] = y_predicted
        ho_score = roc_auc_score(holdout.Response, y_predicted)
        print('# Holdout score is: ' + str(ho_score))
 
    if COMPUTE_TEST_PRED:
        X_tr = train[features]
        y_tr = train.Response
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(test[features])
        #y_predicted = np.argmax(y_pred, axis=1)
        y_predicted = y_pred[:,1]
        test_pred[label] = y_predicted
        return test_pred
    
# want to see the cross-validation results)
COMPUTE_TEST_PRED = True
COMPUTE_HOLDOUT_PRED = True

# Containers for results
oof, test_pred, holdout_pred = {}, {}, {}
######### finish modelling setup ##############
##############################################






##############################################
######### initial modelling ##################
#xgb ~ 1m
xgb_model = xgboost.XGBClassifier(enable_categorical=True, eval_metric='auc', device="cuda")
cross_validate(xgb_model, 'Xgboost_TargetEncoded_Untuned', features=initial_features) 
#0.87968  #0.8823514306751534

#lgb ~ 2m
lgb_model = lightgbm.LGBMClassifier(verbose=-1, eval_metric='auc', device='gpu')
cross_validate(lgb_model, 'LightGBM_TargetEncoded_Untuned', features=initial_features)
#0.87777  #0.8779690873551864

#catboost ~ 3m
catboost_model = catboost.CatBoostClassifier(verbose=False, eval_metric='AUC', task_type='GPU')
cross_validate(catboost_model, 'CatBoost_TargetEncoded_Untuned', features=initial_features)
#0.87712  #0.8777496957149025

#logistic regression ~ 1m
logistic_model = make_pipeline(StandardScaler(),
                      LogisticRegression(penalty='l2', C=0.1, solver='lbfgs'))
cross_validate(logistic_model, 'LogisticRegression', features=initial_features)
#0.86317  #0.863323136866984

#polynomial - ridge
polyridge_model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2),
                      Ridge())
cross_validate(polyridge_model, 'Poly-Ridge untuned', features=initial_features) 
#0.86528


#glm_1
X = StandardScaler().fit_transform(train[initial_features])
X = pd.DataFrame(X, columns=train[initial_features].columns)
X_train, X_test, y_train, y_test = train_test_split(X, train.Response, test_size=0.3, random_state=24)
X_train['Response'] = pd.DataFrame(y_train)
glm_1 = smf.glm(formula = "Response ~ Age + Driving_License + Region_Code + Previously_Insured + Annual_Premium + \
                     Policy_Sales_Channel + Vintage + Gender_encoded + Vehicle_Age_encoded + Vehicle_Damage_encoded", 
                data=X_train,
                family = sm.families.Binomial())
result = glm_1.fit()
print(result.summary())
print(roc_auc_score(y_test, result.predict(X_test))) #0.8365207358593787

#glm_2
#X = StandardScaler().fit_transform(train[initial_features])
#X = pd.DataFrame(X, columns=train[initial_features].columns)
X_train, X_test, y_train, y_test = train_test_split(train, train.Response, test_size=0.3, random_state=24)
X_train['Response'] = pd.DataFrame(y_train)
glm_2 = smf.glm(formula = "Response ~ Age + Driving_License + Previously_Insured + Annual_Premium + \
                Vehicle_Age_encoded + Vehicle_Damage_encoded", 
                data=X_train,
                family = sm.families.Binomial())
result = glm_2.fit()
print(result.summary())
print(roc_auc_score(y_test, result.predict(X_test))) #0.8346020582920359

#glm_3
#X = StandardScaler().fit_transform(train[initial_features])
#X = pd.DataFrame(X, columns=train[initial_features].columns)
X_train, X_test, y_train, y_test = train_test_split(train, train.Response, test_size=0.3, random_state=24)
X_train['Response'] = pd.DataFrame(y_train)
glm_3 = smf.glm(formula = "Response ~ Age + Previously_Insured + Vehicle_Damage_encoded + \
                Policy_Sales_Channel + Vehicle_Age_encoded", 
                data=X_train,
                family = sm.families.Binomial())
result = glm_3.fit()
print(result.summary())
print(roc_auc_score(y_test, result.predict(X_test))) #0.8352272864307876

#glm_lasso
X = StandardScaler().fit_transform(train[initial_features])
X = pd.DataFrame(X, columns=train[initial_features].columns)
X_train, X_test, y_train, y_test = train_test_split(X, train.Response, test_size=0.3, random_state=24)
for alpha in np.arange(0.01, 1.01, 0.01):
    lasso_model = LogisticRegression(C=alpha, penalty='l1', solver='saga', n_jobs=4)
    lasso_model.fit(X_train, y_train)
    #print('L1 alpha: %f' % lasso_model.C, end='\n')
    score = roc_auc_score(y_test, lasso_model.predict(X_test))
    print('L1 alpha: %f' % alpha + ' and AUC is: %f' % score)
######### finish intial models ###############
##############################################



##############################################
######### feature probing ####################
shallow_tree = DecisionTreeClassifier(max_depth=3)
shallow_tree.fit(train[initial_features], train.Response);

plt.figure(figsize=(16, 6))
plot_tree(shallow_tree, feature_names=initial_features, #class_names=label_encoder.classes_, 
          fontsize=7, impurity=True, filled=True, ax=plt.gca())
plt.show()
#1.Vehicle_Damage_encoded
#2.Age
#3.Previously_Insured

train['random'] = np.random.rand(len(train))
full_tree = DecisionTreeClassifier(max_depth=len(train.columns))
full_tree.fit(train[initial_features+['random']], train.Response)
importance_df = pd.DataFrame({'feature': initial_features+['random'], 
                              'importance': full_tree.feature_importances_})
importance_df = importance_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.show()
#4.Policy_Sales_Channel
#5.Vintage
#6.Vehicle_Age_encoded
#7.Annual_Premium
#8.Region_Code
#9.Gender_encoded
#10.random
#11.Driving_License

######### end of feature probing #############
##############################################




##############################################
######### feature probing - EDA ##############
sns.kdeplot(data=train, x='Vehicle_Damage_encoded', hue='Response');
sns.countplot(data=train, x='Response', hue='Vehicle_Damage_encoded') #train[train.Vehicle_Damage_encoded==0].Response.value_counts(normalize=True)

sns.kdeplot(data=train, x='Age', hue='Response');
sns.countplot(data=train, x='Response', hue='Age')

sns.kdeplot(data=train, x='Previously_Insured', hue='Response');
sns.countplot(data=train, x='Response', hue='Previously_Insured')

sns.kdeplot(data=train, x='Policy_Sales_Channel', hue='Response');
sns.countplot(data=train, x='Response', hue='Policy_Sales_Channel')

sns.kdeplot(data=train, x='Vintage', hue='Response');
sns.countplot(data=train, x='Response', hue='Vintage')

sns.kdeplot(data=train, x='Vehicle_Age_encoded', hue='Response');
sns.countplot(data=train, x='Response', hue='Vehicle_Age_encoded')


sns.kdeplot(data=train, x='Annual_Premium', hue='Response');
sns.kdeplot(data=train, x='Region_Code', hue='Response');

sns.kdeplot(data=train, x='Gender_encoded', hue='Response');
sns.countplot(data=train, x='Response', hue='Gender_encoded')

sns.kdeplot(data=train, x='Driving_License', hue='Response');
sns.kdeplot(data=train, x='random', hue='Response');
######### End of feature probing EDA #########
##############################################




##############################################
######### target encoding tuning #############
#split the data for mean encodings
trn, tst = train_test_split(train, test_size=0.3, stratify=train.Response , random_state = 8)


#1.Policy_Sales_Channel
smooth_values = []
score_values = []
for smooth in range(80000, 90001, 1000): #80000 #0.8756888949274835
    trn_, tst_ = target_encode(trn['Policy_Sales_Channel'], 
                               tst['Policy_Sales_Channel'], 
                               target=trn.Response, 
                               min_samples_leaf=round(len(trn)*0.05),
                               smoothing=smooth,
                               noise_level=0)
    trn['Policy_Sales_Channel_TargetEncoded'] = trn_
    tst['Policy_Sales_Channel_TargetEncoded'] = tst_
    
    model = xgboost.XGBRegressor(enable_categorical=True, eval_metric='auc', device="cuda")
    features = ['Age',
     'Driving_License',
     'Region_Code',
     'Previously_Insured',
     'Annual_Premium',
     'Policy_Sales_Channel_TargetEncoded',
     'Vintage',
     'Gender_encoded',
     'Vehicle_Age_encoded',
     'Vehicle_Damage_encoded']
    X_train, X_val, y_train, y_val = train_test_split(trn[features], trn.Response, test_size=0.3, random_state = 64)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = roc_auc_score(y_val, y_pred)
    
    smooth_values.append(smooth)
    score_values.append(score)
    print('Smooth factor is ' + str(smooth) + ' and AUC is: ' + str(score))
    
smooth_Policy_Sales_Channel = pd.DataFrame({'smoothing': smooth_values, 'auc_score': score_values})


#2.Vintage
smooth_values = []
score_values = []
for smooth in range(850000, 1250001, 500): #950000  #0.878571
    trn_, tst_ = target_encode(trn['Vintage'], 
                               tst['Vintage'], 
                               target=trn.Response, 
                               min_samples_leaf=round(len(trn)*0.05),
                               smoothing=smooth,
                               noise_level=0)
    trn['Vintage_TargetEncoded'] = trn_
    tst['Vintage_TargetEncoded'] = tst_
    
    model = xgboost.XGBRegressor(enable_categorical=True, eval_metric='auc', device="cuda")
    features = ['Age',
     'Driving_License',
     'Region_Code',
     'Previously_Insured',
     'Annual_Premium',
     'Policy_Sales_Channel',
     'Vintage_TargetEncoded',
     'Gender_encoded',
     'Vehicle_Age_encoded',
     'Vehicle_Damage_encoded']
    X_train, X_val, y_train, y_val = train_test_split(trn[features], trn.Response, test_size=0.3, random_state = 24)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = roc_auc_score(y_val, y_pred)
    
    smooth_values.append(smooth)
    score_values.append(score)
    print('Smooth factor is ' + str(smooth) + ' and AUC is: ' + str(score))
    
smooth_Vintage_TargetEncoded = pd.DataFrame({'smoothing': smooth_values, 'auc_score': score_values})
######### end of target hyper tuning #########
##############################################




##############################################
######### hyper parameter tuning #############

#1.catboost
def objective(trial):
    X_train, X_valid, y_train, y_valid = train_test_split(train[initial_features], train.Response, test_size=0.3)

    param = {
             "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
             #"objective": "CrossEntropy",
             "iterations": trial.suggest_int("iterations", 800, 2000),
             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
             #"depth": trial.suggest_int("depth", 3, 10),
             "depth": trial.suggest_int("depth", 7, 10),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 90000, 200000),
             "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
             #"boosting_type": "Plain",
             "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
             #"bootstrap_type": "MVS",
             "used_ram_limit": "48gb",
             "eval_metric": 'AUC',
             "task_type": 'CPU'
            }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.5, 1)

    gbm = catboost.CatBoostClassifier(**param)

    gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=1, early_stopping_rounds=200)

    #y_preds = gbm.predict(X_valid)
    y_preds = gbm.predict_proba(X_valid)[:,1]
    #pred_labels = np.rint(preds)
    score = roc_auc_score(y_valid, y_preds)
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=3600)
#study_summaries = optuna.study.get_all_study_summaries()
#0.8757030169


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

cat_param_1 = {'objective': 'CrossEntropy',
               'iterations': 885,
               'learning_rate': 0.1955587373934623,
               'colsample_bylevel': 0.09351884473074819,
               'depth': 8,
               'min_data_in_leaf': 97678,
               'boosting_type': 'Plain',
               'bootstrap_type': 'MVS'
               }

#catboost_1 ~ 11m
catboost_model_1 = catboost.CatBoostClassifier(**cat_param_1, verbose=False, eval_metric='AUC', task_type='CPU')
cross_validate(catboost_model_1, 'CatBoost_TargetEncoded_Tuned_1', features=initial_features)
#0.87679  #0.8775406252537157

# #catboost_2 ~ 11m
# cat_param_2 = {'objective': 'Logloss',
#                'iterations': 1082,
#                'learning_rate': 0.36463881055623176,
#                'colsample_bylevel': 0.06598974581457834,
#                'depth': 8,
#                'min_data_in_leaf': 120840,
#                'boosting_type': 'Ordered',
#                'bootstrap_type': 'Bayesian',
#                'bagging_temperature': 0.06701905564983957
#                }
# catboost_model_2 = catboost.CatBoostClassifier(**cat_param_2, verbose=False, eval_metric='AUC', task_type='CPU')
# cross_validate(catboost_model_2, 'CatBoost_TargetEncoded_Tuned_2', features=initial_features)
# #0.87679  #0.8775406252537157


#2.xgboost
def objective(trial):
    X_train, X_valid, y_train, y_valid = train_test_split(train[initial_features], train.Response, test_size=0.3)

    param = {
             "objective": "binary:logistic",
             "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
             "max_depth": trial.suggest_int("max_depth", 9, 15),
             "min_child_weight": trial.suggest_float('min_child_weight', 1e-10, 1000, log=True),
             'min_split_loss': trial.suggest_float('min_split_loss', 1e-10, 10000, log=True),
             'subsample': trial.suggest_float('subsample', 0, 1),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
             'max_bin': trial.suggest_int("max_bin", 1024, 65536),
             "eval_metric": 'auc',
             'device': "cuda"
            }

    gbm = xgboost.XGBClassifier(**param)

    gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=1)

    #y_preds = gbm.predict(X_valid)
    y_preds = gbm.predict_proba(X_valid)[:,1]
    #pred_labels = np.rint(preds)
    score = roc_auc_score(y_valid, y_preds)
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10000, timeout=18000)



xgb_params_1 = {'max_depth': 13, 
                'min_child_weight': 5,
                'learning_rate': 0.02,
                'colsample_bytree': 0.6,         
                'max_bin': 3000, 
                'n_estimators': 1500}
#xgboost_1 ~ 6m
xgboost_model_1 = xgboost.XGBClassifier(**xgb_params_1, eval_metric='auc', device='cuda')
cross_validate(xgboost_model_1, 'Xgboost_TargetEncoded_Tuned_1', features=initial_features)
#0.88520  #0.9123850795500784


xgb_params_2 = {'n_estimators': 1556,
                'learning_rate': 0.03878492771098787,
                'max_depth': 13,
                'min_child_weight': 1.3380655776726657e-07,
                'min_split_loss': 0.00043944185672855084,
                'subsample': 0.595602072224943,
                'colsample_bytree': 0.2964563512323981,
                'max_bin': 14519}
#xgboost_2 ~ 12m
xgboost_model_2 = xgboost.XGBClassifier(**xgb_params_2, eval_metric='auc', device='cuda')
cross_validate(xgboost_model_2, 'Xgboost_TargetEncoded_Tuned_2', features=initial_features)
#0.88738  #0.9123850795500784


