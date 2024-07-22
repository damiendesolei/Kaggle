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


import statsmodels.api as sm
import statsmodels.formula.api as smf


import matplotlib.pyplot as plt
import datetime
from colorama import Fore, Style



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

#categorical feature distribution
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
train['Gender_encoded'] = train['Gender'].apply(int_encode_gender)
test['Gender_encoded'] = test['Gender'].apply(int_encode_gender)
test[['Gender','Gender_encoded']].drop_duplicates() #show encoding



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
train['Vehicle_Age_encoded'] = train['Vehicle_Age'].apply(int_encode_vehicle_age)
test['Vehicle_Age_encoded'] = test['Vehicle_Age'].apply(int_encode_vehicle_age)
test[['Vehicle_Age','Vehicle_Age_encoded']].drop_duplicates() #show encoding



#integer encode the Vehicle_Damage
def int_encode_vehicle_damage(col):
    if col == 'No':
        return 0
    elif col == 'Yes':
        return 1
    else:
        return 9999
    
#train.groupby('Gender').count()
train['Vehicle_Damage_encoded'] = train['Vehicle_Damage'].apply(int_encode_vehicle_damage)
test['Vehicle_Damage_encoded'] = test['Vehicle_Damage'].apply(int_encode_vehicle_damage)
test[['Vehicle_Damage','Vehicle_Damage_encoded']].drop_duplicates() #show encoding
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
drop_list = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

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
##############################################
######### finish dropping columns ############






##############################################
######### modelling setup  ###################
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

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
        y_pred = model.predict(X_va)
        #y_pred = model.predict_proba(X_va)
        #y_predicted = np.argmax(y_pred, axis=1) #find the highest probability row array position 
        

        va_score = roc_auc_score(y_va, y_pred)
        tr_score = roc_auc_score(y_tr, model.predict(X_tr))
        #tr_score = roc_auc_score(y_tr, np.argmax(model.predict(X_tr), axis=1))
        print(f"# Fold {fold}: tr_auc={tr_score:.5f}, val_auc={va_score:.5f}")

        va_scores.append(va_score)
        tr_scores.append(tr_score)
        #oof_preds[idx_va] = y_predicted #each iteration will fill in 1/5 of the index
        oof_preds[idx_va] = y_pred
            
    elapsed_time = datetime.datetime.now() - start_time
    print(f"{Fore.RED}# Overall val={np.array(va_scores).mean():.5f} {label}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    print(f"{Fore.RED}# {label} Fitting started from {start_time}")
    oof[label] = oof_preds

    if COMPUTE_TEST_PRED:
        X_tr = train[features]
        y_tr = train.Response
        model.fit(X_tr, y_tr)
        y_pred = model.predict(test[features])
        #y_predicted = np.argmax(y_pred, axis=1)
        test_pred[label] = y_pred
        #return test_pred
    
    # if COMPUTE_HOLDOUT_PRED:
    #     X_tr = train[features]
    #     y_tr = train.Response
    #     model.fit(X_tr, y_tr)
    #     y_pred = model.predict_proba(holdout[features])
    #     #y_predicted = np.argmax(y_pred, axis=1)
    #     holdout_pred[label] = y_pred
    #     #return holdout_pred
 

# want to see the cross-validation results)
COMPUTE_TEST_PRED = True
#COMPUTE_HOLDOUT_PRED = True

# Containers for results
oof, test_pred, holdout_pred = {}, {}, {}
######### finish modelling setup ##############
##############################################






##############################################
######### initial modelling ##################
#xgb ~ 4m
xgb_model = xgboost.XGBRegressor(enable_categorical=True, eval_metric='auc', device="cuda")
cross_validate(xgb_model, 'Xgboost untuned', features=initial_features) 
#0.87601

#lgb ~ 3m
lgb_model = lightgbm.LGBMRegressor(verbose=-1, eval_metric='auc', device='gpu')
cross_validate(lgb_model, 'LightGBM untuned', features=initial_features)

#catboost ~ 37m
catboost_model = catboost.CatBoostRegressor(verbose=False, eval_metric='AUC')#, task_type='GPU')
cross_validate(catboost_model, 'CatBoost untuned', features=initial_features)

#logistic regression
logistic_model = make_pipeline(StandardScaler(),
                      LogisticRegression(penalty='l2', C=0.1, solver='lbfgs'))
cross_validate(logistic_model, 'LogisticRegression', features=['Vehicle_Damage_encoded','Previously_Insured'])

#polynomial - ridge
polyridge_model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2),
                      Ridge())
cross_validate(polyridge_model, 'Poly-Ridge untuned', features=initial_features) 
#0.84586


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