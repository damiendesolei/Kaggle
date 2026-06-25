# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pathlib import Path
from collections import defaultdict
import gc, warnings, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)

RNG = np.random.default_rng(42)

N_FOLDS = 5
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100