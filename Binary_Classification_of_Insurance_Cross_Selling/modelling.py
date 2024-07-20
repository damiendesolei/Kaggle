# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
pd.set_option('display.max_columns', None)


##############################################
######### read in data #######################
PATH = r'G:\\kaggle\Binary_Classification_of_Insurance_Cross_Selling\\'

train = pd.read_csv(PATH + 'train.csv', index_col='id', low_memory=True)
test = pd.read_csv(PATH + 'test.csv', index_col='id', low_memory=True)
######### finish reading in data# ############
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
train[['Gender','Gender_encoded']].drop_duplicates() #show encoding



#integer encode the Vehicle_Age
def int_encode_vehicle_age(col):
    if col == '1-2 Year':
        return 0
    elif col == '< 1 Year':
        return 1
    elif col == '> 2 Years':
        return 2  
    else:
        return 9999
    
#train.groupby('Gender').count()
train['Vehicle_Age_encoded'] = train['Vehicle_Age'].apply(int_encode_vehicle_age)
train[['Vehicle_Age','Vehicle_Age_encoded']].drop_duplicates() #show encoding



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
train[['Vehicle_Damage','Vehicle_Damage_encoded']].drop_duplicates() #show encoding
######### finish encoding columns ############
##############################################




##############################################
######### drop certain columns ###############