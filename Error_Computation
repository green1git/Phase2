import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Load the data
launch_data = pd.read_excel('Launch data.xlsx', skiprows=22)
launch_data = launch_data.iloc[:251, :] #Only 15 Setting Data Launches. (27 Removed as do not have constants for this yet)


# These constants will change this is with all Setting ID's
constants = pd.read_csv('all_constants.csv')

import numpy as np
from scipy.integrate import odeint

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

# Modified code to only print a message when constants are not found, regardless of the index
results = []

for _, row in launch_data.iterrows():
    ball_name = row['Name'].replace('.csv', '')
    ball_constants_df = constants[constants['BallName'] == ball_name]

    # Check if constants are found for the ball
    if not ball_constants_df.empty:
        ball_constants = ball_constants_df.iloc[0, 1:].values
        distance, time = compute_distance_and_time(row, ball_constants)
        results.append((ball_name, distance, time))
    else:
        print(f"Constants not found for ball {ball_name}")

# Extracting only the carry from the simulation results
simulated_results = [result[1] for result in results]

# Adding the simulated results to the original launch data DataFrame
launch_data['Predicted Carry'] = pd.Series(simulated_results)

# Calculating the error by subtracting the actual carry from the predicted carry
launch_data['Error'] = launch_data['Predicted Carry'] - launch_data['Carry (yards)']

# Extracting the error data for the histogram
error_data = launch_data['Error'].dropna()

# Creating the error histogram
plt.figure(figsize=(10, 6))
plt.hist(error_data, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Carry Distance Errors')
plt.xlabel('Error (yards)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
