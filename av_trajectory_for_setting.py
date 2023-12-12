import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('coefficients.csv') 

p = 1.293  # kg/m^3
A = 0.0014335  # m^2
m_b = 0.0459  # kg
g = 9.81  # m/s^2
Omega = 2e-5
r = 0.0427  # m

# Initial conditions
x0, y0 = 0, 0
Vx0, Vy0 = 76.95, 13.5
omega0 = 42 * 2 * np.pi  # Convert rev/s to rad/s

# Time points
t = np.linspace(0, 10, 1000)  # 1000 time points between 0 and 10 seconds

# calculate trajectory for a given set of coefficients
def calculate_trajectory(Cd, Cl):
    # System of differential equations
    def model(state, t):
        x, y, Vx, Vy, omega = state

        V = np.sqrt(Vx**2 + Vy**2)

        dxdt = Vx
        dydt = Vy
        dVxdt = -(p * A * V / (2 * m_b)) * (Vx * Cd + Vy * Cl)
        dVydt = (p * A * V / (2 * m_b)) * (Vx * Cl - Vy * Cd) - g
        domegadt = (Omega * omega * V) / r

        return [dxdt, dydt, dVxdt, dVydt, domegadt]

    initial_state = [x0, y0, Vx0, Vy0, omega0]

    # solve
    solution = odeint(model, initial_state, t)

    return solution[:, 0], solution[:, 1]

# Calculate trajectories for each set of coefficients
trajectories = []
for index, row in df.iterrows():
    Cd = row['Cd']  
    Cl = row['Cl']
    x_traj, y_traj = calculate_trajectory(Cd, Cl)
    trajectories.append((x_traj, y_traj))

# average trajectory
average_x = np.mean([traj[0] for traj in trajectories], axis=0)
average_y = np.mean([traj[1] for traj in trajectories], axis=0)

plt.figure(figsize=(8, 6))
plt.plot(average_x, average_y, label='Average Trajectory')
plt.title('Average Golf Ball Trajectory')
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.legend()
plt.grid(True)
plt.show()


