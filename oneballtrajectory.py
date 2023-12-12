import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
p = 1.293  # kg/m^3
A = 0.0014335  # m^2
m_b = 0.0459  # kg
c_d = 0.27
c_l = 0.24
g = 9.81  # m/s^2
Omega = 2e-5
r = 0.0427  # m

# Initial conditions, edit these for each setting
x0, y0 = 0, 0
Vx0, Vy0 = 76.2438, 13.4438
omega0 = 42 * 2 * np.pi  # Convert rev/s to rad/s

# Time points
t = np.linspace(0, 10, 1000)  # 1000 time points between 0 and 10 seconds

# System of differential equations
def model(state, t):
    x, y, Vx, Vy, omega = state

    V = np.sqrt(Vx**2 + Vy**2)

    dxdt = Vx
    dydt = Vy
    dVxdt = -(p * A * V / (2 * m_b)) * (Vx * c_d + Vy * c_l)
    dVydt = (p * A * V / (2 * m_b)) * (Vx * c_l - Vy * c_d) - g
    domegadt = (Omega * omega * V) / r

    return [dxdt, dydt, dVxdt, dVydt, domegadt]

# Initial state
initial_state = [x0, y0, Vx0, Vy0, omega0]

# Solve the system of equations
solution = odeint(model, initial_state, t)

# Extracting results
x_solution, y_solution, Vx_solution, Vy_solution, omega_solution = (
    solution[:, 0],
    solution[:, 1],
    solution[:, 2],
    solution[:, 3],
    solution[:, 4],
)

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_solution, y_solution, label='Trajectory')
plt.title('Golf Ball Trajectory')
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.legend()
plt.grid(True)
plt.show()
