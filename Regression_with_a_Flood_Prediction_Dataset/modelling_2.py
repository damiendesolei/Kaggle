# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:11:37 2024

@author: damie
"""
import time
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold

from xgboost import XGBRegressor
from xgboost import plot_importance

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


path = 'G:\kaggle\Regression_with_a_Flood_Prediction_Dataset\\'

train = pd.read_csv(path + 'train.csv', low_memory=True)
test = pd.read_csv(path + 'test.csv', low_memory=True)



#integet encode the Sex
# def int_encode_sex(col):
#     if col == 'I':
#         return 0
#     elif col == 'F':
#         return 1
#     elif col == 'M':
#         return 3
#     else:
#         return 4
    
  
# train.groupby('Sex').count()
# train['sex'] = train['Sex'].apply(int_encode_sex)
# train[['Sex','sex']].drop_duplicates()


# test['sex'] = test['Sex'].apply(int_encode_sex)
# test[['Sex','sex']].drop_duplicates()


#set up random column
np.random.seed(24)
train['random'] = np.random.rand(train.shape[0])



#train test split
y = train['FloodProbability']
#drop unwanted columns
train.drop(['FloodProbability'], axis=1, inplace=True)



X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state = 24)


#setup metrics 
# y_true = [3, -0.5, 2, 7]
# y_pred = [1, 0.0, 2, 5]
# r2_score(y_true, y_pred)
#r2_score_cv = make_scorer(r2_score, greater_is_better=True)



#starting parameters
param_test ={    
            'eta': sp_uniform(loc=0.0, scale=3.0),
            'max_depth': sp_randint(1, 10),
            'min_child_weight': sp_uniform(loc=0, scale=5000.0),
            #'subsample': sp_uniform(loc=0.2, scale=0.8), 
            'gamma': sp_uniform(loc=0, scale=4.0), 
            #'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'alpha' : sp_uniform(loc=0.0, scale=2000.0),
            'lambda': sp_uniform(loc=0.0, scale=2000.0)
            #'scale_pos_weight': sp_uniform(loc=0, scale=1.0)
            }



n_points_to_test = 10000

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 50000 define only the absolute maximum
xgb_reg = XGBRegressor(device="cuda", n_jobs=1, random_state=24, booster='gbtree', objective='reg:squarederror', #eval_metric=r2_score_cv,
                       n_estimators=50000, early_stopping_rounds=100)

kf = KFold(n_splits=5, shuffle = True, random_state = 1024)
random_search = RandomizedSearchCV(estimator=xgb_reg, param_distributions=param_test, n_iter=n_points_to_test, scoring='r2', 
                                   cv=kf.split(X_train,y_train), refit=True, random_state=24, verbose=1, n_jobs=5)

random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])

best_params = random_search.best_params_
results = pd.DataFrame(random_search.cv_results_)
results.to_csv(path + 'xgb-random-grid-search-results-02.csv', index=False)



#steup cv fold
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

#master model
def train_model(X=X_train, X_test=X_val, y=y_train, params=None, folds=folds, plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        model = xgb.xgbMRegressor(**params, n_estimators = 50000, n_jobs = 5)
        model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                verbose=10000, early_stopping_rounds=200)
        
        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
            
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X.columns
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'xgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('xgb Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction