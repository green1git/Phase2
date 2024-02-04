#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:01:39 2024

@author: milesweedon
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
#from sklearn.feature_selection import SelectFromModel

file_path = "../27 Setting Data/output.csv"
#file_path = "output.csv"

df = pd.read_csv(file_path, header=None)

M = np.array(df)

M = np.transpose(M)

y = np.array([i for i in range(len(M))])

# Use XGBoost
xgb = XGBClassifier()

# Fit model
xgb.fit(M, y)

# Number of features to select
n_features = 10  # Change this to your desired number of features

# Get feature importances and sort them. Sorted in ascending order of importance
importances = xgb.feature_importances_
indices = np.argsort(importances) + 1 # Index at 0, so add 1

# Select the 'n' features to remove
selected_indices = indices[:n_features]

print(selected_indices)
