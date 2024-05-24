# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:11:37 2024

@author: damie
"""

import pandas as pd
import numpy as np
import datetime
from colorama import Fore, Style
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from sklearn.decomposition import PCA
#import umap

from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer, OneHotEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.manifold import TSNE

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import pickle

from sklearn.linear_model import Ridge, LinearRegression

from xgboost import XGBRegressor
from xgboost import plot_importance
import catboost
import lightgbm



path = 'G:\kaggle\Regression_with_a_Flood_Prediction_Dataset\\'
#path = r'C:\Users\damie\Downloads\playground-series-s4e5\\'

train = pd.read_csv(path + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(path + 'test.csv', index_col='id', low_memory=True)

initial_features = list(test.columns)



#EDA
#target distribution
train['FloodProbability'].describe()
plt.hist(train['FloodProbability'], bins=np.linspace(0.2825, 0.7275, 90), density=True)
plt.ylabel('density')
plt.xlabel('FloodProbability')
plt.show()


#train vs test feature distribution
_, axs = plt.subplots(5, 4, figsize=(12, 12))
for col, ax in zip(initial_features, axs.ravel()):
    vc = train[col].value_counts() / len(train)
    ax.bar(vc.index, vc)
    vc = test[col].value_counts() / len(test)
    ax.bar(vc.index, vc, alpha=0.6)
    ax.set_title(col)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
plt.tight_layout()
plt.show()


#correlation matrix
corr_features = initial_features + ['FloodProbability']
cc = np.corrcoef(train[corr_features], rowvar=False)
#plt.figure(figsize=(15, 15))
sns.heatmap(cc, center=0, cmap='coolwarm', annot=True, fmt='.1f',
            xticklabels=corr_features, yticklabels=corr_features)
plt.title('Correlation matrix')
plt.show()


#PCA
pca = PCA()
pca.fit(train[initial_features])
plt.figure(figsize=(3, 2.5))
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
plt.title('Principal Components Analysis')
plt.xlabel('component#')
plt.ylabel('explained variance ratio')
plt.yticks([0, 1])
plt.show()



# Unsupervised UMAP
def plot_embedding(embedding, target, title):
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=1,
        c=target,
        cmap='coolwarm'
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=18)
    plt.show()

train_sample = train.sample(10000)
reducer = umap.UMAP()
plot_embedding(reducer.fit_transform(train_sample[initial_features]),
               train_sample.FloodProbability,
               'Unsupervised UMAP projection of the training dataset')



# Supervised UMAP with regression target
train_sample = train.sample(100000)
reducer = umap.UMAP(n_neighbors=100, target_metric='manhattan',
                    target_weight=0.6, min_dist=1)
plot_embedding(reducer.fit_transform(train_sample[initial_features],
                                     y=train_sample.FloodProbability),
               train_sample.FloodProbability,
               'Supervised UMAP projection of the training dataset')


# t-SNE
train_sample = train.sample(20000)
reducer = TSNE()
plot_embedding(reducer.fit_transform(train_sample[initial_features]),
               train_sample.FloodProbability,
               '(Unsupervised) t-SNE projection of the training dataset')

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


kf = KFold(n_splits=5, shuffle=True, random_state=24)
SINGLE_FOLD = False

def cross_validate(model, label, features=initial_features, n_repeats=1):
    """Compute out-of-fold and test predictions for a given model.
    
    Out-of-fold and test predictions are stored in the global variables
    oof and test_pred, respectively.
    
    If n_repeats > 1, the model is trained several times with different seeds.
    """
    start_time = datetime.datetime.now()
    scores = []
    oof_preds = np.full_like(train.FloodProbability, np.nan, dtype=float)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train)):
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train.iloc[idx_tr].FloodProbability
        y_va = train.iloc[idx_va].FloodProbability
        
        y_pred = np.zeros_like(y_va, dtype=float)
        for i in range(n_repeats):
            m = clone(model)
            if n_repeats > 1:
                mm = m
                if isinstance(mm, Pipeline):
                    mm = mm[-1]
                mm.set_params(random_state=i)
            m.fit(X_tr, y_tr)
            y_pred += m.predict(X_va)
        y_pred /= n_repeats
        
#         residuals = y_va - y_pred
#         plt.figure(figsize=(6, 2))
#         plt.scatter(y_pred, residuals, s=1)
#         plt.axhline(0, color='k')
#         plt.show()
        
        score = r2_score(y_va, y_pred)
        print(f"# Fold {fold}: R2={score:.5f}")
        scores.append(score)
        oof_preds[idx_va] = y_pred
        if SINGLE_FOLD: break
            
    elapsed_time = datetime.datetime.now() - start_time
    print(f"{Fore.GREEN}# Overall: {np.array(scores).mean():.5f} {label}"
          f"{' single fold' if SINGLE_FOLD else ''}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    oof[label] = oof_preds
    
    if COMPUTE_TEST_PRED:
        # Retrain n_repeats times with the whole dataset and average
        y_pred = np.zeros(len(test), dtype=float)
        X_tr = train[features]
        y_tr = train.FloodProbability
        for i in range(n_repeats):
            m = clone(model)
            if n_repeats > 1:
                mm = m
                if isinstance(mm, Pipeline):
                    mm = mm[-1]
                if isinstance(mm, TransformedTargetRegressor):
                    mm = mm.regressor
                mm.set_params(random_state=i)
            m.fit(X_tr, y_tr)
            y_pred += m.predict(test[features])
        y_pred /= n_repeats
        test_pred[label] = y_pred



# want to see the cross-validation results)
COMPUTE_TEST_PRED = True

# Containers for results
oof, test_pred = {}, {}

#linear models
model = make_pipeline(StandardScaler(),
                      LinearRegression())
cross_validate(model, 'LinearRegression')

#polynomial - ridge
model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2),
                      Ridge())
cross_validate(model, 'Poly-Ridge')

#splie - ridge
model = make_pipeline(StandardScaler(),
                      SplineTransformer(),
                      Ridge())
cross_validate(model, 'Spline-Ridge')


# # Nystroem transformer + ridge
# model = make_pipeline(StandardScaler(),
#                       Nystroem(n_components=600),
#                       Ridge())
# cross_validate(model, 'Nystroem-Ridge')


#stats model - linear regression
import statsmodels.api as sm
X = sm.add_constant(train[initial_features])
res = sm.OLS(train.FloodProbability, X, missing='error').fit()
res.summary()


# XGBoost
xgb_params = {'grow_policy': 'depthwise'
              ,'n_estimators': 100
              ,'learning_rate': 0.2639887908316703
              ,'max_depth': 10
              ,'reg_lambda': 62.46661785864016
              ,'min_child_weight': 0.33652299514909034
              ,'colsample_bytree': 0.2319730052165745
              ,'objective': 'reg:squarederror'
              ,'tree_method': 'hist'
              ,'max_bin': 2048
              ,'gamma': 0} # 0.83868
model = XGBRegressor(**xgb_params)
cross_validate(model, 'XGBoost')



#catboost
model = catboost.CatBoostRegressor(verbose=False)
cross_validate(model, 'CatBoost')


#lgb
model = lightgbm.LGBMRegressor(verbose=-1)
cross_validate(model, 'LightGBM')



#look at the sum of each rows, and summaize by the sum vs probability
temp = train.FloodProbability.groupby(train[initial_features].sum(axis=1)).mean()
plt.scatter(temp.index, temp, s=1, c=(temp.index.isin(np.arange(72, 76))), cmap='coolwarm')
plt.xlabel('sum of twenty initial features')
plt.ylabel('mean flood probability')
plt.show()

# Add the special1 and fsum features
for df in [train, test]:
    df['fsum'] = df[initial_features].sum(axis=1) # for tree models
    df['special1'] = df['fsum'].isin(np.arange(72, 76)) # for linear models


#refit the models 
#linear regression
model = make_pipeline(StandardScaler(),
                      LinearRegression())
cross_validate(model, 'LinearRegression_special1', features=initial_features+['special1'])

#polynomial - ridge
model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2),
                      Ridge())
cross_validate(model, 'Poly-Ridge_special1', features=initial_features+['special1'])

#splie - ridge
model = make_pipeline(StandardScaler(),
                      SplineTransformer(),
                      Ridge())
cross_validate(model, 'Spline-Ridge_special1', features=initial_features+['special1'])

#ridge - onehot factor sum
model = make_pipeline(OneHotEncoder(categories=[np.unique(train.fsum)],
                                    drop='first', sparse_output=False),
                      StandardScaler(),
                      Ridge())
cross_validate(model, 'Ridge one-hot fsum', features=['fsum'])

#catboost factor sum
model = catboost.CatBoostRegressor(verbose=False)
cross_validate(model, 'CatBoost_fsum', features=initial_features+['fsum'])


#Evaluation model results
result_list = []
for label in oof.keys():
    mask = np.isfinite(oof[label])
    score = r2_score(train.FloodProbability[mask], oof[label][mask])
    result_list.append((label, score))
result_df = pd.DataFrame(result_list, columns=['label', 'score'])
result_df.sort_values('score', inplace=True, ascending=False)

plt.figure(figsize=(12, len(result_df) * 0.4 + 0.4))
bars = plt.barh(np.arange(len(result_df)), result_df.score, color='lightgreen')
plt.annotate('Best model without feature engineering', 
             (0.850, list(result_df.label).index('CatBoost')),
             xytext=(0.855, list(result_df.label).index('CatBoost')+2),
             arrowprops={'width': 2, 'color': 'darkgreen'},
             color='darkgreen')
plt.gca().bar_label(bars, fmt='%.5f')
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.gca().invert_yaxis()
plt.xlim(0.835, 0.875)
plt.xlabel(f'{"fold 0" if SINGLE_FOLD else "5-fold cv"} r2 score (higher is better)')
plt.show()


#submission 
submission = pd.DataFrame(test_pred)
submission['FloodProbability'] = (submission['CatBoost_fsum']
                                  +submission['Ridge one-hot fsum']
                                  +submission['Poly-Ridge_special1']
                                  +submission['Spline-Ridge_special1']
                                  +submission['LinearRegression_special1'])/5
submission['id'] = test.index
submission = submission[['id','FloodProbability']]
#create submission file
submission.to_csv(path+'submission_20240525_1.csv', index=False)




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
        'eta': sp_uniform(loc=0.0, scale=3.0),
        'min_child_weight':  sp_uniform(loc=0, scale=100),
        'gamma': sp_uniform(loc=0, scale=20.0),
        'alpha' : sp_uniform(loc=0.0, scale=100.0),
        'lambda': sp_uniform(loc=0.0, scale=100.0),
        'subsample': [0.6,0.8,1],
        'colsample_bytree': [0.6,0.8,1],
        'max_depth': sp_randint(1, 8)
        }

#xgb = XGBRegressor(device="cuda", n_estimators=200, objective='reg:squarederror')
xgb = XGBRegressor(n_estimators=200, objective='reg:squarederror')


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
print('\n Best R2 for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
best_params = random_search.best_params_
results = pd.DataFrame(random_search.cv_results_)
results.to_csv(path + 'xgb-random-grid-search-results-01.csv', index=False)



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
