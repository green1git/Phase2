#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:40:56 2024

@author: milesweedon
"""

'''
All points, not averages
'''

import os
import pandas as pd
import numpy as np

# Set the folder path to the current working directory
folder_path = '.'

# Initialise M
M = None


# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'): #Filter by the csvs
        if filename != "output.csv": #Ignore any pre-existing outputs
            file_path = os.path.join(folder_path, filename)
    
            # Read the CSV file
            #print(file_path)
            df = pd.read_csv(file_path, header=None) #No header 
    
            # Extract the last four columns
            df_last_four = df.iloc[:, -4:]
            df_last_four[4] = np.arange(1, 28)
            current_array = np.array(df_last_four)
            
            # Save to matrix M
            if M is None:
                M = current_array
            else:
                M = np.vstack((M, current_array))


# Save output as output.csv
np.savetxt("AllData.csv", M, delimiter=",")
