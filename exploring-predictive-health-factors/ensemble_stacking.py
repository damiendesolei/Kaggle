# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:59:09 2025

@author: zrj-desktop
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


PATH = r'G:\\kaggle\exploring-predictive-health-factors\model\ensemble\\'


# Read in model predictions
# PCOS_Single_XGB_4 - Version 7
pred_xgb = pd.read_csv(PATH + 'xgb_12_parameters_reserved_cv_auc_0.8587.csv')
# PCOS_Single_CAT_2 - Version 4
pred_cat = pd.read_csv(PATH + 'cat_12_parameters_reserved_cv_auc_0.8683.csv')
# PCOS_Single_LGB_5 - Version 4
pred_lgb = pd.read_csv(PATH + 'lgb_12_parameters_reserved_cv_auc_0.8560.csv')



# Define independent variables (exclude intercept)
X = pd.DataFrame()
X['pred_xgb'] = pred_xgb['pred_xgb']
X['pred_cat'] = pred_cat['pred_lgb'] #typo to fix
X['pred_lgb'] = pred_lgb['pred_lgb']
y = pred_xgb['PCOS']

# Add an intercept column
X = sm.add_constant(X)  # Adds a column of 1s for the intercept

# Fit a GLM with a logit link function and NO intercept
glm_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
glm_fitted = glm_model.fit()




# Read in submission files
submission_xgb = pd.read_csv(PATH + 'xgb_12_parameters_submission_cv_auc_0.8587_LB_0.916.csv')
submission_cgb = pd.read_csv(PATH + 'cat_12_parameters_submission_cv_auc_0.8683_LB_0.903.csv')
submission_lgb = pd.read_csv(PATH + 'lgb_12_parameters_submission_cv_auc_0.8560_LB_0.923.csv')

X_ = pd.DataFrame()
X_['pred_xgb'] = submission_xgb['PCOS']
X_['pred_cat'] = submission_cgb['PCOS'] 
X_['pred_lgb'] = submission_lgb['PCOS']
# Add an intercept column
X_ = sm.add_constant(X_)  # Adds a column of 1s for the intercept

y_submission = glm_fitted.predict(X_)


# Ensemble
submission = submission_lgb.copy()
submission['PCOS'] = y_submission
submission.to_csv(PATH+'lgb_0.923_xgb_0.916_cat_0.903_blend_1.csv', index=False)


