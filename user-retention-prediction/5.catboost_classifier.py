# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:36:46 2025

@author: zrj-desktop
"""
# https://www.kaggle.com/code/dc5e964768ef56302a32/5-catboost-classifier

import polars as pl
import numpy as np


PATH = r'G:\\kaggle\user-retention-prediction\\'

train=pl.read_csv(f"{PATH}train.csv").with_columns([
    ((1540051199-pl.col("Timestamp"))//(24*60*60)).alias("days")
])
sub=pl.read_csv(f"{PATH}submit_sample.csv")


last_time=train.group_by("ID").agg((1540051199-pl.col("Timestamp").max()).alias("lasttime"))

sub_last_time=sub.join(last_time,on="ID",how="left").fill_null(2*24*3600)
sub_last_time["lasttime"].max()/24/3600

#分析发现评测的用户都是前一日有登陆的用户，所以建立线下验证集的时候选取的ID也得都是前一日有过登陆的用户
#由于label需要统计未来七天的数据，所以我准备使用days=7的有记录的ID作为用户集，
#使用days=0-6的数据计算label,使用days=7-13作为统计特征的数据集。


def create_sample(shift_days):
    
    train_id=train.filter(pl.col("days")==shift_days)[["ID"]].unique()
    
    train_features=train.filter(pl.col("ID").is_in(train_id["ID"].to_list())
                               ).filter(pl.col("days").is_in([shift_days+i for i in range(5)])).with_columns(
        pl.col("days")-shift_days,
        ((1540051199-pl.col("Timestamp"))%(24*60*60)).alias("seconds")
        
    )
    if shift_days>0:
        label=train_id.join(
            train.filter(pl.col("days").is_in([shift_days-i for i in range(1,8)])).group_by(["ID"]).agg([
            pl.col("days").n_unique().cast(int).alias("label")
        ]),on="ID",how="left").fill_null(0)
        dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
        thresholds=dist["ratio"].to_list()
    else:
        label=None
        thresholds=[0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
                             0.07644968, 0.12103906, 0.46647806]

    return train_features, label, thresholds
    #return sub,label
    
    

train_features, train_label, train_thresholds = create_sample(14)
valid_features, valid_label, valid_thresholds = create_sample(7)




def create_features(train_features):
    train_id=train_features[["ID"]].unique()
    for k in range(7):
        train_features_temp=train_features.filter(pl.col("days")==k)
        features=train_features_temp.group_by("ID").agg([
        *[(pl.col("ActionType")==n).sum().alias(f"ActionType_{n}_{k}") for n in range(5)],
        pl.col("seconds").min().alias(f"seconds_min_{k}"),
        pl.col("seconds").max().alias(f"seconds_max_{k}"),
        pl.col("seconds").count().alias(f"seconds_count_{k}"),
    ])
        train_id=train_id.join(features,on="ID",how="left")
    return train_id
train_data=create_features(train_features)
train_data=train_data.join(train_label,on="ID",how="left")

valid_data=create_features(valid_features)
valid_data=valid_data.join(valid_label,on="ID",how="left")

train_data.shape, valid_data.shape


features_name=[i for i in train_data.columns if i not in ["ID","label"]]







# Hyper parameter tuning
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import catboost as cb
from catboost import CatBoostClassifier,CatBoostRegressor,CatBoostRanker
import optuna
import gc
from scipy.special import softmax

def multiclass_logloss(y_actual, y_pred):
    """
    y_actual: array of shape [n_samples], integer class labels
    y_pred: array of shape [n_samples, n_classes], raw logits
    """
    # Convert raw logits to probabilities using softmax
    prob = softmax(y_pred, axis=1)

    # Clip to prevent log(0)
    eps = 1e-15
    prob = np.clip(prob, eps, 1 - eps)

    # Pick the predicted probability of the correct class
    correct_log_probs = -np.log(prob[np.arange(len(y_actual)), y_actual])

    return np.mean(correct_log_probs)


STUDY = False
if STUDY:
    
    FEATURES = features_name
    train = train_data.to_pandas()
    
    # def reduce_mem_usage(df, float16_as32=True):
    #     #memory_usage() to sum all ram for each column, B->KB->MB
    #     start_mem = df.memory_usage().sum() / 1024**2
    #     print('Memory usage of current dataframe is {:.2f} MB'.format(start_mem))
    #     non_date_columns = [col for col in df.columns if df[col].dtype != 'datetime64[ns]']
    
    #     for col in non_date_columns:
    #         col_type = df[col].dtype 
    #         if col_type != object and str(col_type)!='category':#不是object也就是说这里处理的是数值类型的变量
    #             c_min,c_max = df[col].min(),df[col].max() #求出这列的最大值和最小值
    #             if str(col_type)[:3] == 'int':#如果是int类型的变量,不管是int8,int16,int32还是int64
    #                 #如果这列的取值范围是在int8的取值范围内,那就对类型进行转换 (-128 到 127)
    #                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
    #                     df[col] = df[col].astype(np.int8)
    #                 #如果这列的取值范围是在int16的取值范围内,那就对类型进行转换(-32,768 到 32,767)
    #                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
    #                     df[col] = df[col].astype(np.int16)
    #                 #如果这列的取值范围是在int32的取值范围内,那就对类型进行转换(-2,147,483,648到2,147,483,647)
    #                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
    #                     df[col] = df[col].astype(np.int32)
    #                 #如果这列的取值范围是在int64的取值范围内,那就对类型进行转换(-9,223,372,036,854,775,808到9,223,372,036,854,775,807)
    #                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
    #                     df[col] = df[col].astype(np.int64)  
    #             else:#如果是浮点数类型.
    #                 #如果数值在float16的取值范围内,如果觉得需要更高精度可以考虑float32
    #                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
    #                     if float16_as32:#如果数据需要更高的精度可以选择float32
    #                         df[col] = df[col].astype(np.float32)
    #                     else:
    #                         df[col] = df[col].astype(np.float16)  
    #                 #如果数值在float32的取值范围内，对它进行类型转换
    #                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
    #                     df[col] = df[col].astype(np.float32)
    #                 #如果数值在float64的取值范围内，对它进行类型转换
    #                 else:
    #                     df[col] = df[col].astype(np.float64)
    #     #current ram
    #     end_mem = df.memory_usage().sum() / 1024**2
    #     print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #     #ram reduction%
    #     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    #     return df
    
    # train = reduce_mem_usage(train)
    
    # Define the parameter space
    def objective(trial):
        
        param = {
            'loss_function': 'MultiClass', 
            #'grow_policy': 'Lossguide',
            'task_type': 'GPU', 
            #'devices':'0',
            #'used_ram_limit': '8gb',
            'gpu_ram_part':0.8,
            #'bootstrap_type': 'Bernoulli',  
    
            'iterations': 2000,
            #'iterations': trial.suggest_int('iterations', 1000, 3000, step=200),
            'depth': trial.suggest_int('depth', 12, 16, step=1),  
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 128, 256, step=4),
            
            #'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.6, 1.0), # Random Subspace Method (rsm not supported on GPU)
            #'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 10, log=True),
            #'random_strength': trial.suggest_float('random_strength', 0.001, 0.1, log=True),
    
            'random_seed': 2025,
            'logging_level': "Silent"  # Suppress CatBoost logs
        }
    
        # Time series cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=False)
        scores = []
        
        for i, (train_index, test_index) in enumerate(skf.split(train, train["label"])):       
            #x_train = train.loc[train_index,FEATURES].copy()
            x_train = train.loc[train_index,FEATURES].copy()
            y_train = train.loc[train_index,"label"]
            x_valid = train.loc[test_index,FEATURES].copy()
            y_valid = train.loc[test_index,"label"]
            #x_test = test[FEATURES].copy()
    
    
            # Create CatBoost pools
            dtrain = cb.Pool(x_train, label=y_train)#, cat_features=CATS)
            dvalid = cb.Pool(x_valid, label=y_valid)#, cat_features=CATS)
    
            # Train Catboost model
            model = cb.CatBoostClassifier(**param)
            model = model.fit(
                dtrain,
                eval_set=dvalid,
                early_stopping_rounds=100,
                use_best_model=True
            )
            # Predict on validation set
            y_pred = model.predict_proba(dvalid)
        
            logloss = multiclass_logloss(y_valid, y_pred)  # WMAE for regression
            scores.append(logloss)  
        
        mean_logloss = np.mean(scores)
        
        # Cleanup
        del model
        gc.collect()
        
        return mean_logloss
    
    
    # Run Optuna study
    N_HOUR = 4
    CORES = 1
    
    print("Start running hyper parameter tuning..")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=3600*N_HOUR, n_jobs=CORES, catch=(MemoryError,))  # 3600*n hour
    
    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best logloss:", study.best_value)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Format the file name with the best score
    OUT_PATH = r'G:\\kaggle\user-retention-prediction\model\\'
    file_name = f"Cbt_MultiClass_Logloss_{best_score:.6f}.csv"
    
    # Save the best parameters to a CSV file
    df_param = pd.DataFrame([best_params])  # Convert to DataFrame
    df_param.to_csv(OUT_PATH+file_name, index=False)  # Save to CSV
    
    print(f"Best parameters saved to {file_name}")






########################################################
params={
    'iterations': 2000,
    'loss_function': 'MultiClass',
    'learning_rate': 0.012173458515288418,
    'depth': 15,
    'min_data_in_leaf': 140,
    'l2_leaf_reg': 0.6147877031247082,
    #'random_strength': 0.006167528225817892,
    
    'verbose': 100,
    #'eval_metric':"SMAPE",
    'early_stopping_rounds': 100,
    'task_type': 'GPU',
    }
model = CatBoostClassifier(**params)
model.fit(train_data[features_name].to_numpy(), train_data["label"].to_numpy(),
#          eval_set=(valid_data[features_name].to_numpy(), valid_data["label"].to_numpy())
         )


pred = model.predict_proba(valid_data[features_name].to_numpy())



from scipy.optimize import minimize, basinhopping
def post_processing(pred,thresholds):
    n_classes = 8

    def adjust_prob(prob, weights):
        adjusted_prob = prob * weights
        return adjusted_prob / adjusted_prob.sum(axis=1, keepdims=True)

    def objective(weights):
        adjusted_prob = adjust_prob(pred, weights)
        adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1),minlength=n_classes)/len(adjusted_prob)
        return np.sum((adjusted_distribution - np.array(thresholds))**2)

    initial_weights = np.ones(n_classes)
    bounds = [(1e-6, None) for _ in range(n_classes)]

    result = minimize(objective, initial_weights, bounds=bounds,method='Powell')
    optimal_weights = result.x
    adjusted_prob = adjust_prob(pred, optimal_weights)
    return adjusted_prob


adjusted_prob=post_processing(pred, valid_thresholds)

adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1))/len(adjusted_prob)
print("actual distribution:", valid_thresholds)
print("adjusted distribution:", adjusted_distribution)


sub=valid_data[["ID","label"]].to_pandas()
sub["pred"]=np.argmax(adjusted_prob,axis=1)


#转成回归再后处理
sub=valid_data[["ID","label"]].to_pandas()
sub["pred"]=pred@np.array([0,1,2,3,4,5,6,7])
sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], np.cumsum(valid_thresholds)).clip(0,7)


score = (200*abs(sub["pred"]-sub["label"])/(sub["pred"]+sub["label"])).fillna(0).mean()


test_features, test_label, test_thresholds=create_sample(0)
test_data=create_features(test_features)
pred=model.predict_proba(test_data[features_name].to_numpy())
adjusted_prob=post_processing(pred,test_thresholds)
sub=test_data[["ID"]].to_pandas()
sub["pred"]=np.argmax(adjusted_prob,axis=1)

adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1))/len(adjusted_prob)
print("actual distribution:", test_thresholds)
print("adjusted distribution:", adjusted_distribution)


#转成回归再后处理
sub=test_data[["ID"]].to_pandas()

sub["pred"]=pred@np.array([0,1,2,3,4,5,6,7])
OUT_PATH = r'G:\\kaggle\user-retention-prediction\model\\'
sub_name = f"Cbt_MultiClass_raw_smape_{score:.6f}.csv"
sub.to_csv(OUT_PATH+sub_name, index=False)

sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], np.cumsum(test_thresholds)).clip(0,7)

OUT_PATH = r'G:\\kaggle\user-retention-prediction\model\\'
sub_name = f"Cbt_MultiClass_smape_{score:.6f}.csv"
sub.to_csv(OUT_PATH+sub_name, index=False)