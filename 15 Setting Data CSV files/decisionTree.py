#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:21:59 2024

@author: milesweedon
"""
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

file_path = "./output.csv"

df = pd.read_csv(file_path, header=None)

M = np.array(df)

M = np.transpose(M)

y = np.array([i+1 for i in range(len(M))])


# Assume X is your feature matrix and y are the labels
clf = RandomForestClassifier(n_estimators=5000)
clf = clf.fit(M, y)

# Selecting features based on importance
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(M)  # X_new contains only the important features
'''


'''

num_runs = 100
feature_counts = np.zeros(M.shape[1]) # A shape of 15

for _ in range(num_runs):
    # Create a bootstrap sample (or use other subsampling methods)
    sample_indices = np.random.choice(range(M.shape[0]), size=int(M.shape[0]*0.8))
    #print(sample_indices)
    M_sample = M[sample_indices]
    y_sample = y[sample_indices]

    # Feature selection on the subsample
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(M_sample, y_sample)
    model = SelectFromModel(clf, prefit=True)
    selected_features = model.get_support()

    # Update counts
    feature_counts += selected_features

# Determine stable features
stability_threshold = num_runs * 0.7 # e.g., selected in 70% of runs
stable_features = np.where(feature_counts >= stability_threshold)[0]

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

file_path = "./output.csv"
df = pd.read_csv(file_path, header=None)
M = np.array(df).T
y = np.arange(len(M)) + 1

num_runs = 10000
total_importance = np.zeros(M.shape[1])  # Initialize to zero

for _ in range(num_runs):
    clf = RandomForestClassifier(n_estimators=5000)
    clf.fit(M, y)
    total_importance += clf.feature_importances_

# Average the feature importances
average_importance = total_importance / num_runs

# Sort the features by importance
sorted_features = np.argsort(average_importance)[::-1]  # Descending order

# Now, `sorted_features` contains indices of features sorted by their importance
