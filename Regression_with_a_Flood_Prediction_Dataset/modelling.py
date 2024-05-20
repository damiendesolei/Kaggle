# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:11:37 2024

@author: damie
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from xgboost import XGBRegressor
from xgboost import plot_importance



#path = 'G:\kaggle\Regression_with_a_Flood_Prediction_Dataset\\'
path = r'C:\Users\damie\Downloads\playground-series-s4e5\\'

train = pd.read_csv(path + 'train.csv', low_memory=True)
test = pd.read_csv(path + 'test.csv', low_memory=True)



#EDA
plt.hist(train['FloodProbability'] , density=True)
plt.ylabel('density')
plt.xlabel('FloodProbability')
plt.show()


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
#np.random.seed(24)
#train['random'] = np.random.rand(train.shape[0])



#train test split
y = train['FloodProbability']
#drop unwanted columns
train.drop(['FloodProbability','id'], axis=1, inplace=True)



X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state = 24)


#setup metrics 
# y_true = [3, -0.5, 2, 7]
# y_pred = [1, 0.0, 2, 5]
# r2_score(y_true, y_pred)
r2_score_cv = make_scorer(r2_score, greater_is_better=True)



#xgb hyper parameter tuning
params = {
        'eta': sp_uniform(loc=0.0, scale=3.0),
        'min_child_weight':  sp_uniform(loc=0, scale=500),
        'gamma': sp_uniform(loc=0, scale=20.0),
        'alpha' : sp_uniform(loc=0.0, scale=500.0),
        'lambda': sp_uniform(loc=0.0, scale=500.0),
        'subsample': [0.6,0.8,1],
        'colsample_bytree': [0.6,0.8,1],
        'max_depth': sp_randint(1, 8)
        }

#xgb = XGBRegressor(device="cuda", n_estimators=200, objective='reg:squarederror')
xgb = XGBRegressor(n_estimators=200, objective='reg:squarederror')


folds = 5
param_comb = 1000

kf = KFold(n_splits=folds, shuffle = True, random_state = 1024)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=r2_score_cv, n_jobs=4, 
                                   cv=kf.split(X_train,y_train), verbose=3, random_state=1024)


random_search.fit(X_train, y_train)


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best R2 for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
best_params = random_search.best_params_
results = pd.DataFrame(random_search.cv_results_)
results.to_csv(path + 'xgb-random-grid-search-results-00.csv', index=False)



#fit the model
xgb_model = XGBRegressor(n_estimators=200
                         ,objective='reg:squarederror'
                         ,learning_rate=0.3
                         ,max_depth=2
                         ,min_child_weight=1
                         ,gamma=1
                         ,subsample=0.8
                         ,colsample_bytree=0.8
                         ,random_state=1024)
xgb_model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)], 
             verbose=False)

y_pred = xgb_model.predict(X_val)
r2_score(y_val, y_pred) #0.8051867396014364

plot_importance(xgb_model, max_num_features=25, importance_type='weight', xlabel='weight')
plot_importance(xgb_model, max_num_features=25, importance_type='gain', xlabel='gain')



#validation_0-rmsle:0.15438
#LB top: 0.14482



#predict the test set
X_test = test.drop(['id'], axis=1)
y_test = xgb_model.predict(X_test)


submission = pd.DataFrame(test['id'])
submission['FloodProbability'] = y_test

#create submission file
submission.to_csv(path+'submission_20240520_1.csv', index=False)
