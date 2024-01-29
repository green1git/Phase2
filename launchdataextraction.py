#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:21:59 2024

@author: milesweedon
"""

import pandas as pd
import numpy as np

file_path = "./LaunchData.csv"

df = pd.read_csv(file_path)

array = df.iloc[22:, 1:6]

nparray = np.array(array)
nparray = np.concatenate((nparray, np.zeros([len(nparray), 1])), axis = 1)
    
for item in range(len(nparray)):
    string = str(nparray[item, 0]) + str(nparray[item, 1]) + str(nparray[item, 2])
    nparray[item, -1] = int(string)
    

nparray = nparray[nparray[:, -1].argsort()]


# Use the unique labels to filter
labels = np.unique(nparray[:, -1])

carry1 = []

for item in range(len(nparray)):
    if nparray[item, -1] == labels[0]:
        carry1.append(float(nparray[item, -2]))
        
carry1 = np.array(carry1)

print(np.mean(carry1))
print(np.std(carry1))