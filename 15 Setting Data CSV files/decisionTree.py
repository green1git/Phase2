#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:21:59 2024

@author: milesweedon
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

file_path = "./output.csv"

df = pd.read_csv(file_path, header=None)

M = np.array(df)

M = np.transpose(M)

y = np.array([i+1 for i in range(len(M))])

# Assume X is your feature matrix and y are the labels
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(M, y)

# Selecting features based on importance
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(M)  # X_new contains only the important features

