# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:52:59 2024

@author: zrj-desktop
"""

import datetime
from colorama import Fore, Style
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import MaxNLocator

from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier, plot_tree

import xgboost
import catboost
import lightgbm

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import scipy.stats


#path = 'G:\kaggle\Classification_with_an_Academic_Success_Dataset\\'
path = r'C:\Users\damie\Downloads\playground-series-s4e6\\'

train = pd.read_csv(path + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(path + 'test.csv', index_col='id', low_memory=True)
train.info()

#set aside holdout sets for fitting ensemble weight
train, holdout = train_test_split(train, test_size=0.1, stratify=train.Target , random_state = 24)


#assigne correct type to columns from the organizer
initial_features = list(test.columns)
cat_features = ['Marital status', 'Application mode', 'Course',
                'Previous qualification', 'Nacionality', "Mother's qualification", 
                "Father's qualification", "Mother's occupation",
                "Father's occupation"]
num_features = [f for f in train._get_numeric_data() if (f not in ['Target']) and (f not in cat_features)]
print(f'Numeric cols: {len(num_features)}')
print(f'Categorical cols: {len(cat_features)}')

for feature in cat_features:
    for df in [train, test, holdout]:
        df[feature] = df[feature].astype('category')


#lable encode the target
label_encoder = LabelEncoder()
label_encoder.fit(train.Target)
#show fited encoder
mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
for label, index in mapping.items():
    print(f"{index} -> {label}")
#encode
train_targets = label_encoder.transform(train.Target)
holdout_targets = label_encoder.transform(holdout.Target)


    
    
    
    
    

###############################################################################
#feature exploration
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(train[initial_features], train.Target);


plt.figure(figsize=(16, 6))
plot_tree(dt, feature_names=initial_features, class_names=label_encoder.classes_, fontsize=7, impurity=True, filled=True, ax=plt.gca())
plt.show()


importance_df = pd.DataFrame({'feature': initial_features, 'importance': dt.feature_importances_})
importance_df = importance_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.show()


#plot categorical features
plt.figure(figsize=(18, 24))
plotnumber = 1
# Loop through each column
for col in cat_features:
    if plotnumber <= len(cat_features):
        plt.subplot(4, 2, plotnumber)
        ax = sns.countplot(x=train[col], hue=train['Target'], palette='bright')
        
    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables by Target', fontsize=40, y=1)
#plt.tight_layout()
plt.show()




#plot numeric features
plt.figure(figsize=(18, 44))
plotnumber = 1
for column in num_features:
    if plotnumber <= len(num_features):
        ax = plt.subplot(11, 3, plotnumber)
        sns.kdeplot(train[column], color='deepskyblue', fill=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        plt.xlabel(column)
        ax.grid(False)
        
    plotnumber += 1

plt.suptitle('Distribution of Numeric Variables', fontsize=40, y=1)
plt.tight_layout()
plt.show()


###############################################################################

#integer encode: Target column
# encode_target = {
#     'Graduate': 0,
#     'Dropout': 1,
#     'Enrolled': 2
# }
# train['Target'] = train['Target'].map(encode_target)

###############################################################################

#plot categorical features vs target
cat_cols = [f for f in train.columns if (train[f].dtype != 'O' and train[f].nunique() <100) or (train[f].dtype == 'O' and f not in ['Target']) ]
custom_palette = (0.2, 0.9, 0.6), 'crimson', (0.8, 0.5, 0.3)
for col in cat_cols:
    contingency_table = pd.crosstab(train[col], train['Target'], normalize='index')
    sns.set(style="whitegrid")
    contingency_table.plot(kind="bar", stacked=True, color=custom_palette,figsize=(20, 8))
    plt.title(f"Percentage Distribution of Target across {col}")
    plt.xlabel(col)
    plt.ylabel("Percentage")
    plt.legend(title="Target Class")
    plt.show()



#target distribution
#train['Target'].describe()
plt.hist(train['Target'],facecolor='skyblue', edgecolor='white')
plt.ylabel('Count')
plt.xlabel('Target')
plt.show()


#train vs test feature distribution
_, axs = plt.subplots(12, 3, figsize=(12, 36))
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
plt.figure(figsize=(20, 16))
plt.title('Correlation Matrix\n')
sns.heatmap(train[initial_features].corr(), vmin=-1, vmax=1, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.show()



#modelling
#kf = KFold(n_splits=5, shuffle=True, random_state=24)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

def cross_validate(model, label, features=initial_features):
    """Compute out-of-fold and test predictions for a given model.
    
    Out-of-fold and test predictions are stored in the global variables
    oof and test_pred, respectively.
    
    If n_repeats > 1, the model is trained several times with different seeds.
    """
    start_time = datetime.datetime.now()
    tr_scores = []
    va_scores = []
    oof_preds = np.full_like(train_targets, np.nan, dtype=np.float64)
    #for fold, (idx_tr, idx_va) in enumerate(kf.split(train)):
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.Target)):    
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train_targets[idx_tr]
        y_va = train_targets[idx_va]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_va)
        y_predicted = np.argmax(y_pred, axis=1) #find the highest probability row array position 
        

        va_score = accuracy_score(y_va, y_predicted)
        tr_score = accuracy_score(y_tr, np.argmax(model.predict_proba(X_tr), axis=1))
        print(f"# Fold {fold}: tr_accuracy={tr_score:.5f}, val_accuracy={va_score:.5f}")

        va_scores.append(va_score)
        tr_scores.append(tr_score)
        oof_preds[idx_va] = y_predicted #each iteration will fill in 1/5 of the index
            
    elapsed_time = datetime.datetime.now() - start_time
    print(f"{Fore.RED}# Overall val={np.array(va_scores).mean():.5f} {label}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    print(f"{Fore.RED}# {label} Fitting started from {start_time}")
    oof[label] = oof_preds

    if COMPUTE_TEST_PRED:
        X_tr = train[features]
        y_tr = train_targets
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(test[features])
        y_predicted = np.argmax(y_pred, axis=1)
        test_pred[label] = y_predicted
        #return test_pred
    
    if COMPUTE_HOLDOUT_PRED:
        X_tr = train[features]
        y_tr = train_targets
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(holdout[features])
        y_predicted = np.argmax(y_pred, axis=1)
        holdout_pred[label] = y_predicted
        #return holdout_pred
 

# want to see the cross-validation results)
COMPUTE_TEST_PRED = True
COMPUTE_HOLDOUT_PRED = True

# Containers for results
oof, test_pred, holdout_pred = {}, {}, {}





#xgb
xgb_model = xgboost.XGBClassifier(enable_categorical=True)
cross_validate(xgb_model, 'Xgboost untuned', features=initial_features)

feat_importances = pd.Series(xgb_model.feature_importances_, index=initial_features)
plt.figure(figsize=(7, 8))  # Set the figure size (width, height)
feat_importances.nlargest(50).plot(kind='barh').invert_yaxis()



#lgb
model = lightgbm.LGBMClassifier(verbose=-1)
cross_validate(model, 'Lightboost untuned', features=initial_features)


#catboost
model = catboost.CatBoostClassifier(cat_features=cat_features,verbose=False)
cross_validate(model, 'Catboost untuned', features=initial_features)




#Evaluation model results
result_list = []
for label in oof.keys():
    mask = np.isfinite(oof[label])
    score = accuracy_score(targets[mask], oof[label][mask])
    result_list.append((label, score))
result_df = pd.DataFrame(result_list, columns=['label', 'score'])
result_df.sort_values('score', inplace=True, ascending=False)

plt.figure(figsize=(12, len(result_df) * 0.4 + 0.4))
bars = plt.barh(np.arange(len(result_df)), result_df.score, color='lightgreen')
# plt.annotate('Best model without feature engineering', 
#              (0.850, list(result_df.label).index('CatBoost')),
#              xytext=(0.855, list(result_df.label).index('CatBoost')+2),
#              arrowprops={'width': 2, 'color': 'darkgreen'},
#              color='darkgreen')
plt.gca().bar_label(bars, fmt='%.5f')
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.gca().invert_yaxis()
plt.xlim(0.825, 0.875)
SINGLE_FOLD = False
plt.xlabel(f'{"fold 0" if SINGLE_FOLD else "5-fold cv"} accuracy_score (higher is better)')
plt.show()





#submission 
submission = pd.DataFrame(test_pred)
submission['Target'] = label_encoder.inverse_transform(submission['Catboost untuned'])


submission['id'] = test.index
submission = submission[['id','Target']]
#create submission file
submission.to_csv(path+'submission_20240602_1.csv', index=False)





#xgb hyper parameter tuning
params = {
        'eta': sp_uniform(loc=0.0, scale=2.0),
        'min_child_weight':  sp_uniform(loc=0, scale=100),
        'gamma': sp_uniform(loc=0, scale=2.0),
        'alpha' : sp_uniform(loc=0.0, scale=1.0),
        'lambda': sp_uniform(loc=0.0, scale=1.0),
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': sp_randint(2, 7)
        }

xgb = xgboost.XGBClassifier(device="cuda", n_estimators=200)


folds = 3
param_comb = 10000

kf = KFold(n_splits=folds, shuffle = True, random_state = 1024)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=r2_score_cv, n_jobs=3, 
                                   cv=kf.split(X_train,y_train), verbose=0, random_state=1024)


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
results.to_csv(path + 'xgb-random-grid-search-results-02.csv', index=False)





#catboost untuned
model = catboost.CatBoostClassifier(cat_features=cat_features)
y = train['Target']
col = initial_features
X_train, X_val, y_train, y_val = train_test_split(train[col], y, test_size=0.4, random_state = 1024)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)
accuracy_score(y_val, y_pred)
