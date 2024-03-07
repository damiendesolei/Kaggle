# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:10:36 2024

@author: zrj-desktop
"""

import gc

import pandas as pd
pd.set_option('display.max_columns', None)


train_base = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_base.csv')

train_static_0_0 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_static_0_0.csv')
train_static_0_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_static_0_1.csv')
train_static_0 = pd.concat([train_static_0_0, train_static_0_1], axis=0, sort=False)
####check column nan
# percent_missing = train_static_0.isnull().mean() * 100 / len(train_static_0)
# missing_value_df = pd.DataFrame({'column_name': train_static_0.columns,
#                                  'percent_missing': percent_missing})
####remove columns with more than 80% nan
train_static_0 = train_static_0.loc[:, train_static_0.isnull().mean() <= .80]

train_static_cb_0 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_static_cb_0.csv')

train_applprev_1_0 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_applprev_1_0.csv')
train_applprev_1_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_applprev_1_1.csv')
train_applprev_1 = pd.concat([train_applprev_1_0, train_applprev_1_1], axis=0, sort=False)
train_applprev_2 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_applprev_2.csv')

train_static_0 = train_static_0.select_dtypes((int, float))
train_static_cb_0 = train_static_cb_0.select_dtypes((int, float))
train_applprev_1_0 = train_applprev_1_0.select_dtypes((int, float))
train_applprev_1_1 = train_applprev_1_1.select_dtypes((int, float))
train_applprev_2 = train_applprev_2.select_dtypes((int, float))

train_other_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_other_1.csv')
train_tax_registry_a_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_tax_registry_a_1.csv')
train_tax_registry_b_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_tax_registry_b_1.csv')
train_tax_registry_c_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_tax_registry_c_1.csv')

train_credit_bureau_a_1_0 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_1_0.csv')
train_credit_bureau_a_1_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_1_1.csv')
train_credit_bureau_a_1_2 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_1_2.csv')
train_credit_bureau_a_1_3 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_1_3.csv')
train_credit_bureau_b_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_b_1.csv')
train_credit_bureau_a_2_0 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_0.csv')
train_credit_bureau_a_2_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_1.csv')
train_credit_bureau_a_2_2 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_2.csv')
train_credit_bureau_a_2_3 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_3.csv')
train_credit_bureau_a_2_4 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_4.csv')
train_credit_bureau_a_2_5 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_5.csv')
train_credit_bureau_a_2_6 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_6.csv')
train_credit_bureau_a_2_7 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_7.csv')
train_credit_bureau_a_2_8 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_8.csv')
train_credit_bureau_a_2_9 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_9.csv')
train_credit_bureau_a_2_10 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_a_2_10.csv')
train_credit_bureau_b_2 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_credit_bureau_b_2.csv')

train_deposit_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_deposit_1.csv')
train_person_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_person_1.csv')
train_person_2 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_person_2.csv')

train_debitcard_1 = pd.read_csv(r'G:\kaggle\Home_credit\csv_files\train\train_debitcard_1.csv')


X_train = pd.merge(train_base, train_static_0, how='left', on='case_id')
X_train = pd.merge(X_train, train_static_cb_0, how='left', on='case_id')

train_applprev = pd.merge(train_applprev_1, train_applprev_2, how='inner', on=('case_id','num_group1'))
chk = pd.merge(train_applprev_1_0, train_applprev_1_1, how='inner', on='case_id')
