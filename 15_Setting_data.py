#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:12:09 2023

@author: archie
"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
sns.set()

## Note this is for the 15 setting data
## Dictionary containing each datframe containing 15 setting tests ##

# Folder path
folder_path = '/Users/archie/Downloads/OneDrive_1_21-11-2023'


if os.path.exists(folder_path):
    
    # Dictionary to hold different ball tests (E.g Ball-11-PH)
    ball_tests_dict = {}

    # Iterate over each file in the zip
    for file_name in os.listdir(folder_path):
        
        file_path = os.path.join(folder_path, file_name)

        
        df = pd.read_csv(file_path)


        ball_tests_dict[file_name] = df
else:
    ball_tests_dict


## New column names ##

new_column_names = [
    'Ball Name', 'Date of Test', 'Setting ID', 'Ball number', 
    'Ball Size (inches)', 'Reynolds Number', 'Spin Ratio', 
    'Coefficient of Drag', 'Coefficient of Lift'
]

# Iterate through the dictionaryapplying new_column_names
for key, df in ball_tests_dict.items():
    
    original_column_names = df.columns.tolist()
    
    df.columns = new_column_names

    new_row = pd.DataFrame([original_column_names], columns=new_column_names)

    ball_tests_dict[key] = pd.concat([new_row, df], ignore_index=True)



## Dictionaries to hold different orientation data ##
ball_tests_dict_pp = {}
ball_tests_dict_ph = {}

for key, value in ball_tests_dict.items():
    if 'PP' in key:
        ball_tests_dict_pp[key] = value
    elif 'PH' in key:
        ball_tests_dict_ph[key] = value



## Visulaising how drag and lift coefficents change for different setting conditions ##

# Generating 15 distinct colors from a colormap
cmap = plt.cm.get_cmap('viridis', 15) 

plt.figure(figsize=(14, 8))

def plot_data(df, orientation_label):
    for setting_id in range(1, 16):  # Assuming Setting IDs are from 1 to 15
        setting_data = df[df['Setting ID'] == setting_id]
        for _, row in setting_data.iterrows():
            plt.scatter(row['Coefficient of Drag'], row['Coefficient of Lift'], color=cmap(setting_id-1), label=f'{orientation_label} Setting ID {setting_id}' if setting_id == 1 else "")

# Plotting 'PP' data
for key, df in ball_tests_dict_pp.items():
    plot_data(df, 'PP')

# Plotting 'PH' data
for key, df in ball_tests_dict_ph.items():
    plot_data(df, 'PH')

plt.xlabel('Coefficient of Drag')
plt.ylabel('Coefficient of Lift')
plt.title('Coefficient of Drag vs Lift by Test Conditions')

# Custom legend for 15 settings
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Setting ID {i}',
                              markerfacecolor=cmap(i-1), markersize=10) for i in range(1, 16)]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()

plt.show()



## Visualising Spin ratio change under different Setting ID's ##

plt.figure(figsize=(12, 8))

for key, df in ball_tests_dict.items():
    # Convert 'Spin Ratio' to numeric, coercing any non-numeric values to NaN
    numeric_spin_ratio = pd.to_numeric(df['Spin Ratio'], errors='coerce')

    plt.scatter(df.index + 1, numeric_spin_ratio, label=key, s=10)

plt.xlabel('Setting ID')
plt.ylabel('Spin Ratio')
plt.title('Spin Ratio for Each DataFrame in the Dictionary')
plt.legend()
plt.show()



# Initialize a dictionary to store 'Spin Ratio' values by index
spin_ratios_by_index = {}

# Iterate over each DataFrame and aggregate 'Spin Ratio' values
for key, df in ball_tests_dict.items():
    for index, row in df.iterrows():
        if index not in spin_ratios_by_index:
            spin_ratios_by_index[index] = []
        spin_ratios_by_index[index].append(row['Spin Ratio'])


# Calculate mean and range (extrema) for each index
mean_spin_ratios = []
error_bars = []
for index, ratios in spin_ratios_by_index.items():
    # Convert all values in ratios to numeric, non-numeric values become NaN
    numeric_ratios = pd.to_numeric(ratios, errors='coerce')

    # Compute the mean, ignoring NaN values
    mean_spin_ratios.append(np.nanmean(numeric_ratios))
    
    # Compute the range (max - min), ignoring NaN values
    error_bars.append(np.nanmax(numeric_ratios) - np.nanmin(numeric_ratios))

# Corrected range for indices
indices = range(1, 16)  # Indices from 1 to 15
plt.errorbar(indices, mean_spin_ratios, yerr=error_bars, fmt='ro', ecolor='black', elinewidth=1, capsize=5, markersize=2, label='Average Spin Ratio')

plt.xlabel('Setting ID')
plt.ylabel('Average Spin Ratio')
plt.title('Average Spin Ratio for each Setting ID')
plt.xticks(indices)  # Setting x-ticks to match the indices
plt.legend()
plt.show()

