import pandas as pd
import numpy as np
import statistics as s
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load the data
launch_data = pd.read_excel('Launch data.xlsx', skiprows=22)
launch_data = launch_data.iloc[:251, :] #Only 15 Setting Data Launches. (27 Removed as do not have constants for this yet)

# These constants will change this is with all Setting ID's
constants = pd.read_csv('all_constants.csv')



# Constants common to all simulations
A = 0.0014335  # m^2
m_b = 0.0459  # kg
g = 9.81  # m/s^2
r = 0.02136  # m
Omega = 2e-5
p = 1.1849  # kg/m^3
vis = 0.000015461  # kinematic viscosity of air

# System of differential equations
def model(state, t, constants):
    x, y, Vx, Vy, omega = state
    
    a1, a2, a3, b1, b2, b3, c1, c2, c3, c4, d1, d2 = constants
    V = np.sqrt(Vx**2 + Vy**2)
    R = 2*V*r/vis
    spin_ratio = omega*r/V

    c_l = (a1 + a2/R**5 + a3/R**7) + (b1 + b2*np.log(R)/R**2 + b3/R**2) * spin_ratio
    c_d = (c1 + c2/R**3 + c3/R**5 + c4/R**7) + (d1 + d2*np.log(R)/R**2)*spin_ratio**2
    
    dxdt = Vx
    dydt = Vy
    dVxdt = -(p * A * V / (2 * m_b)) * (Vx * c_d + Vy * c_l)
    dVydt = (p * A * V / (2 * m_b)) * (Vx * c_l - Vy * c_d) - g
    domegadt = -(Omega * omega * V) / r
    
    return [dxdt, dydt, dVxdt, dVydt, domegadt]

# Function to compute distance and time
def compute_distance_and_time(row, constants):
    # Initial conditions
    spinrps = row['Spin Rate (rps)']
    spin_rate = spinrps * 2 * np.pi  # convert to rad/s
    v0ft = row['Ball Speed (ft/s)']
    v0 = v0ft * 0.3048  # convert to m/s
    launchdeg = row['Launch Angle (degrees)']
    launch_angle = launchdeg * np.pi / 180  # convert to radians
    Vx0, Vy0 = v0 * np.cos(launch_angle), v0 * np.sin(launch_angle)
    omega0 = spin_rate  # Convert rev/s to rad/s

    initial_state = [0, 0, Vx0, Vy0, omega0]
    t = np.linspace(0, 10, 1000)  # Time points

    # Solve the system of equations
    solution = odeint(model, initial_state, t, args=(constants,))

    # Extracting results
    x_solution, y_solution = solution[:, 0], solution[:, 1]
    x_solution, y_solution = x_solution[y_solution>=0], y_solution[y_solution>=0]
    
    distance = x_solution[-1] * 1.0936  # Convert to yards
    time = t[len(x_solution) - 1]
    
    return distance, time


#Animation - removal steps
'''
# Step 1: Gather all error data
all_errors = {}  # Initialize as a dictionary to hold errors for each setting

constants_folder_path = 'XG-PCA_constants'  # Update this path as needed
constants_files = os.listdir(constants_folder_path)

for constants_file in constants_files:
    if constants_file.endswith('.csv'):
        setting_id = int(constants_file.split('_')[0])
        
        constants_path = os.path.join(constants_folder_path, constants_file)
        constants = pd.read_csv(constants_path)

        results = []
        for _, row in launch_data.iterrows():
            ball_name = row['Name'].replace('.csv', '')
            ball_constants_df = constants[constants['BallName'] == ball_name]
            if not ball_constants_df.empty:
                ball_constants = ball_constants_df.iloc[0, 1:].values
                distance, time = compute_distance_and_time(row, ball_constants)
                results.append((ball_name, distance, time))
        
        simulated_results = [result[1] for result in results]
        launch_data['Predicted Carry'] = pd.Series(simulated_results)
        
        launch_data['Error'] = launch_data['Predicted Carry'] - launch_data['Carry (yards)']
        error_data = launch_data['Error'].dropna().tolist()  # Convert error data to a list
        
        # Update the dictionary with errors for the current setting
        all_errors[setting_id] = error_data

# Define the bin edges based on the global min and max errors across all settings
global_min_error = min([min(errors) for errors in all_errors.values()])
global_max_error = max([max(errors) for errors in all_errors.values()])
bins = np.linspace(global_min_error, global_max_error, num=30)  # Adjust 'num' for desired bin granularity

histograms_data = {}
for setting_id, errors in all_errors.items():
    histogram, bin_edges = np.histogram(errors, bins=bins)
    histograms_data[setting_id] = histogram


print(histograms_data)



# Step 4: Animate the Histograms
fig, ax = plt.subplots()

def update(frame):
    ax.clear()  # Clear current bars to ensure the plot updates
    # Ensure correct data is used for each frame
    data = histograms_data[frame]  # Access the histogram data for the current frame
    ax.bar(bins[:-1], data, width=np.diff(bins), align='edge', color='royalblue')
    ax.set_xlim([global_min_error, global_max_error])
    ax.set_ylim([0, max(data)*1.1])  # Adjust y-limit to fit the highest bar with some margin
    ax.set_title(f"Number of Setting ID's included: {frame}")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    print(frame)


anim = FuncAnimation(fig, update, frames=sorted(histograms_data.keys(), reverse=True), interval=1000, repeat=False)

plt.show()

'''

# Displaying 1 - line plot
'''
# Path to the folder containing constants files
constants_folder_path = 'XG-PCA_constants'  # Update this path as needed

# Initialize a dictionary to store mean errors with setting IDs as keys
mean_errors_data = {}

constants_files = os.listdir(constants_folder_path)

# Iterate over each constants file
for constants_file in constants_files:
    if constants_file.endswith('.csv'):
        setting_id = int(constants_file.split('_')[0])  # Extract setting ID from file name
        
        # Load the current constants
        constants_path = os.path.join(constants_folder_path, constants_file)
        constants = pd.read_csv(constants_path)

        results = []
        for _, row in launch_data.iterrows():
            ball_name = row['Name'].replace('.csv', '')
            ball_constants_df = constants[constants['BallName'] == ball_name]
            if not ball_constants_df.empty:
                ball_constants = ball_constants_df.iloc[0, 1:].values
                distance, time = compute_distance_and_time(row, ball_constants)
                results.append((ball_name, distance, time))
        
        # Adding the simulated results to the launch data DataFrame
        simulated_results = [result[1] for result in results]
        launch_data['Predicted Carry'] = pd.Series(simulated_results)
        
        # Calculating the error
        launch_data['Error'] = launch_data['Predicted Carry'] - launch_data['Carry (yards)']
        error_data = launch_data['Error'].dropna()
        mean_error = s.mean(abs(error_data))
        
        # Store mean error with setting ID as key
        mean_errors_data[setting_id] = mean_error

        print(f"Mean error for {constants_file}: {mean_error}")

# Plot the errors
setting_ids_with_none = [0] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Generate x values as a range with the correct length
x_values = list(range(len(setting_ids_with_none)))

# Sort mean errors and setting IDs in descending order
sorted_mean_errors = [v for _, v in sorted(mean_errors_data.items(), key=lambda item: item[0], reverse=True)]

plt.figure(figsize=(10, 6))
plt.plot(x_values, sorted_mean_errors, marker='o', linestyle='-', color='blue', label='PCA Removal Order')
plt.xlabel('Number of Setting IDs Removed')
plt.ylabel('Mean Error (yards)')

# Set the x-ticks to the generated x_values and the labels to the setting IDs (with None for the initial state)
plt.xticks(x_values, setting_ids_with_none)

# Draw a horizontal line at mean_errors[0] value for both datasets
base_error = 1.76
plt.axhline(y=base_error, color='r', linestyle='--', label='')

# Move 'Base Error' text to the right, placing it in the middle of the figure
# Assuming the midpoint of the x-axis range for central placement
mid_point = x_values[len(x_values) // 2]
plt.text(mid_point, sorted_mean_errors[0], 'Base Error (1.76 yards)', color='r', va='bottom', ha='center')

plt.grid(True)
plt.legend()
plt.savefig('PCA_removal_error.png', dpi=300)
plt.show()
'''

# Comparing 5  -line plot
'''
# Paths to the folders containing constants files
constants_folder_paths = ['AE_KNN_1', 'AE_KNN_2', 'AE_KNN_3', 'AE_KNN_4', 'AE_KNN_5']

# Initialize a list to store dictionaries for mean errors for each folder
mean_errors_data_list = [{} for _ in range(len(constants_folder_paths))]

# Assuming launch_data is defined somewhere
# launch_data = pd.DataFrame([...])

# Maximum Setting ID assumed based on your requirement (e.g., starting from 15)
max_setting_id = 15

# Iterate over each constants folder and its corresponding mean_errors_data dictionary
for folder_index, constants_folder_path in enumerate(constants_folder_paths):
    constants_files = os.listdir(constants_folder_path)

    for constants_file in constants_files:
        if constants_file.endswith('.csv'):
            # Assuming setting_id from the file name is directly used for ordering
            setting_id = int(constants_file.split('_')[0])

            constants_path = os.path.join(constants_folder_path, constants_file)
            constants = pd.read_csv(constants_path)

            results = []
            for _, row in launch_data.iterrows():
                ball_name = row['Name'].replace('.csv', '')
                ball_constants_df = constants[constants['BallName'] == ball_name]
                if not ball_constants_df.empty:
                    ball_constants = ball_constants_df.iloc[0, 1:].values
                    distance, time = compute_distance_and_time(row, ball_constants)
                    results.append((ball_name, distance, time))

            simulated_results = [result[1] for result in results]
            launch_data['Predicted Carry'] = pd.Series(simulated_results)
            launch_data['Error'] = launch_data['Predicted Carry'] - launch_data['Carry (yards)']
            error_data = launch_data['Error'].dropna()
            mean_error = s.mean(abs(error_data))

            mean_errors_data_list[folder_index][setting_id] = mean_error

# Initialize your plot
plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'purple', 'orange']
labels = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5']

# Generate x values based on the maximum number of settings considered (16 in this case for IDs 0 through 15)
x_values = list(range(max_setting_id + 1))  # 0 through 15

for i, mean_errors_data in enumerate(mean_errors_data_list):
    # Reversed order of setting IDs for plotting, from 15 down to 0
    sorted_mean_errors = [mean_errors_data.get(max_setting_id - id, None) for id in x_values]

    plt.plot(x_values, sorted_mean_errors, marker='o', linestyle='-', color=colors[i], label=labels[i])

plt.xlabel('Number of Setting IDs Removed')
plt.ylabel('Mean Carry Error (yards)')
plt.xticks(range(11))  # Labels from 15 down to 0

base_error = 1.76
plt.axhline(y=base_error, color='r', linestyle='--', label='Base Error: 1.76 yards')

plt.legend()
plt.grid(True)

plt.savefig('corrected_multi_folder_autoencoder_removal_error_descending.png', dpi=300)
plt.show()
'''
#Comparing 2 - line plot 
'''
# Paths to the folders containing constants files
constants_folder_paths = ['XG-PCA_constants', 'AE_KNN_2']  # Update these paths as needed

# Initialize dictionaries to store mean errors with setting IDs as keys
mean_errors_data1 = {}
mean_errors_data2 = {}

# Iterate over each constants folder
for i, constants_folder_path in enumerate(constants_folder_paths):
    constants_files = os.listdir(constants_folder_path)

    # Iterate over each constants file
    for constants_file in constants_files:
        if constants_file.endswith('.csv'):
            setting_id = int(constants_file.split('_')[0])  # Extract setting ID from file name
            
            # Load the current constants
            constants_path = os.path.join(constants_folder_path, constants_file)
            constants = pd.read_csv(constants_path)

            results = []
            for _, row in launch_data.iterrows():
                ball_name = row['Name'].replace('.csv', '')
                ball_constants_df = constants[constants['BallName'] == ball_name]
                if not ball_constants_df.empty:
                    ball_constants = ball_constants_df.iloc[0, 1:].values
                    distance, time = compute_distance_and_time(row, ball_constants)
                    results.append((ball_name, distance, time))
            
            # Adding the simulated results to the launch data DataFrame
            simulated_results = [result[1] for result in results]
            launch_data['Predicted Carry'] = pd.Series(simulated_results)
            
            # Calculating the error
            launch_data['Error'] = launch_data['Predicted Carry'] - launch_data['Carry (yards)']
            error_data = launch_data['Error'].dropna()
            mean_error = s.mean(abs(error_data))
            
            # Store mean error with setting ID as key
            if i == 0:
                mean_errors_data1[setting_id] = mean_error
            else:
                mean_errors_data2[setting_id] = mean_error

            print(f"Mean error for {constants_file}: {mean_error}")

# Plot the errors
setting_ids_with_none = [0] + [1,2,3,4,5,6,7,8,9,10]
x_values = list(range(len(setting_ids_with_none)))

# Sort mean errors and setting IDs in descending order
sorted_mean_errors1 = [v for _, v in sorted(mean_errors_data1.items(), key=lambda item: item[0], reverse=True)]
sorted_mean_errors2 = [v for _, v in sorted(mean_errors_data2.items(), key=lambda item: item[0], reverse=True)]
sorted_setting_ids = sorted(mean_errors_data1.keys(), reverse=True)

plt.figure(figsize=(10, 6))
plt.plot(x_values, sorted_mean_errors1, marker='o', linestyle='-', color='blue', label='XG-BOOST/PCA')
plt.plot(x_values, sorted_mean_errors2, marker='o', linestyle='-', color='green', label='Autoencoder (k=2)')
plt.xlabel('Number of Setting IDs Removed')
plt.ylabel('Mean Error (yards)')

# Draw a horizontal line at mean_errors[0] value for both datasets
base_error = 1.76
plt.axhline(y=base_error, color='r', linestyle='--', label='')

# Move 'Base Error' text to the right, placing it in the middle of the figure
# Assuming the midpoint of the x-axis range for central placement
mid_point = x_values[len(x_values) // 2]
plt.text(mid_point, base_error, f'Base Error ({base_error} yards)', color='r', va='bottom', ha='center')

plt.xticks(x_values)  # Set integer ticks on the x-axis
plt.grid(True)
plt.legend()
plt.savefig('comparison_removal_error.png', dpi=300)
plt.show()
'''