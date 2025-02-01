# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:59:09 2025

@author: zrj-desktop
"""

import pandas as pd
import statsmodels.api as sm


PATH = r'G:\\kaggle\exploring-predictive-health-factors\model\ensemble\\'


# Read in model predictions
# PCOS_Single_XGB_4 - Version 1
pred_xgb = pd.read_csv(PATH + 'xgb_12_parameters_submission_cv_auc_0.8587_LB_0.916.csv')
# cat_12_parameters_submission_cv_auc_0.8421.csv
pred_cat = pd.read_csv(PATH + 'cat_12_parameters_submission_cv_auc_0.8421_LB_0.900.csv')
# PCOS_Single_LGB_2 - Version 2
pred_lgb = pd.read_csv(PATH + 'lgb_12_parameters_submission_cv_auc_0.8227_LB_0.923.csv')



# Define independent variables (exclude intercept)
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Fit a GLM with a logit link function and NO intercept
glm_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
result = glm_model.fit()

# Print coefficients explicitly
print("GLM Coefficients (No Intercept):")
print(result.params)