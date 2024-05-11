# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:11:37 2024

@author: damie
"""

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
r2_score_cv = make_scorer(r2_score, greater_is_better=True)



#xgb hyper parameter tuning
params = {
        'eta': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        'min_child_weight': [1, 3, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5, 10, 20],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [2, 3, 4, 5, 6]
        }

xgb = XGBRegressor(n_estimators=200, objective='binary:logistic', nthread=1)


folds = 5
param_comb = 100

kf = KFold(n_splits=folds, shuffle = True, random_state = 1024)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=r2_score_cv, n_jobs=4, 
                                   cv=kf.split(X_train,y_train), verbose=3, random_state=1024)


random_search.fit(X_train, y_train)


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
best_params = random_search.best_params_
results = pd.DataFrame(random_search.cv_results_)
results.to_csv(path + 'xgb-random-grid-search-results-01.csv', index=False)



#fit the model
xgb_model = XGBRegressor(n_estimators=500
                         ,objective='reg:squaredlogerror'
                         ,learning_rate=0.3
                         ,max_depth=5
                         ,min_child_weight=1
                         ,gamma=0.5
                         ,subsample=0.8
                         ,colsample_bytree=0.8
                         ,random_state=1024)
xgb_model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)], 
             verbose=True)

plot_importance(xgb_model, max_num_features=20, importance_type='weight', xlabel='weight')
plot_importance(xgb_model, max_num_features=20, importance_type='gain', xlabel='gain')

#validation_0-rmsle:0.15438
#LB top: 0.14482
