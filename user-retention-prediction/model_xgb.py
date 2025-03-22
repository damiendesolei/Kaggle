# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:08:21 2025

@author: zrj-desktop
"""

import polars as pl
import numpy as np

PATH = r'G:\\kaggle\user-retention-prediction\\'


train=pl.read_csv(PATH+"train.csv").with_columns([
    ((1540051199-pl.col("Timestamp"))//(24*60*60)).alias("days")
])  # days prior to 2018-10-20 23:59:59
sub=pl.read_csv(PATH+"submit_sample.csv")



last_time=train.group_by("ID").agg((1540051199-pl.col("Timestamp").max()).alias("lasttime"))

sub_last_time=sub.join(last_time,on="ID",how="left").fill_null(2*24*3600)
sub_last_time["lasttime"].max()/24/3600



# Finds IDs that logged in within 7 days and their number of days logged in
label=train.filter(pl.col("days")==7)[["ID"]].unique().join(  
    train.filter(pl.col("days")<7).group_by(["ID"]).agg([
    pl.col("days").n_unique().cast(int).alias("label")
]),on="ID",how="left").fill_null(0)
dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
thresholds=np.cumsum(dist["ratio"].to_list()) 



shift_days=7

# possible features
## average hours online
## max hours online (last action time - first action time)
## max consecutive days logged in
## certain action count
## consecutive days of certain action or combination of actions
## holidays vs weekdays
## login time of the day (e.g. morning, afternoon, evening)


features=train.filter(pl.col("days")==shift_days)[["ID"]].unique().join(
    train.filter(pl.col("days").is_in([shift_days+i for i in range(7)])).group_by(["ID","days"]).agg([
    pl.col("ActionType").count().alias("action_count"),
]),on="ID",how="left").fill_null(0)