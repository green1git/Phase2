import os
import pandas as pd
import numpy as np

# Set the folder path to the current working directory
folder_path = '.'

# Initialize an empty array to store test condition averages
avgs = np.zeros([15, 4])

# Initialise count - used for the averaging later
count = 0


# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'): #Filter by the csvs
        if filename != "output.csv": #Ignore any pre-existing outputs
            file_path = os.path.join(folder_path, filename)
    
            # Read the CSV file
            df = pd.read_csv(file_path, header=None) #No header 
    
            # Extract the last four columns
            df_last_four = np.array(df.iloc[:, -4:])
    
            
            # Append the current reading to the array
            avgs = np.add(avgs, df_last_four)
            
            # Iterate count
            count+=1


# Take the average, using the count variable
avgs = avgs / count

# Print resultr
print("Row-wise average of the final four columns from each CSV file:")
print(avgs)

# Save output as output.csv
np.savetxt("output.csv", avgs, delimiter=",")
