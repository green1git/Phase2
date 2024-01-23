import os
import pandas as pd
import numpy as np

# Set the folder path to the current working directory
folder_path = '.'

# Initialize an empty list to store DataFrames
dataframes = np.zeros([15, 4])

count = 0


# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)

        # Extract the last four columns
        df_last_four = np.array(df.iloc[:, -4:])

        
        # Append the DataFrame to the list
        dataframes = np.add(dataframes, df_last_four)
        
        count+=1


# Concatenate all DataFrames along rows
#concatenated_df = pd.concat(dataframes)

# Compute the row-wise average
#row_wise_average = concatenated_df.mean(axis=1)


row_wise_average = dataframes / count


print("Row-wise average of the final four columns from each CSV file:")
print(row_wise_average)

np.savetxt("output.csv", row_wise_average, delimiter=",")
