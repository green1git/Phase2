# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:06:00 2024

@author: olist
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
p = 1.293  # kg/m^3
A = 0.0014335  # m^2
m_b = 0.0459  # kg
g = 9.81  # m/s^2
r = 0.0427/2  # m
Omega = 2e-5

vis = 0.0000153754 # kinematic viscosity of air

# initial conditions
spin_rate = 42*2*np.pi
v0 = 254*0.3048
launch_angle = 10*np.pi/180 # radians


spin_ratio = spin_rate*r/v0
R = 2*v0*r/vis # Reynolds

# finding cl and cd constants
a1 = 0.0774
a2 = 1
a3 = 1
b1 = 0.9885
b2 = 1
b3 = 1

c1 = 0.2084
c2 = 1
c3 = 1
c4 = 1
d1 = 2.3781
d2 = 1

c_l = (a1 + a2/R**5 + a3/R**7) + (b1 + b2*np.log(R)/R**2 + b3/R**2) * spin_ratio
c_d = (c1 + c2/R**3 + c3/R**5 + c4/R**7) + (d1 + d2*np.log(R)/R**2)*spin_ratio**2

# Initial conditions
x0, y0 = 0, 0
Vx0, Vy0 = v0*np.cos(launch_angle), v0*np.sin(launch_angle)
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

x_solution, y_solution = x_solution[y_solution>=0], y_solution[y_solution>=0]

distance = x_solution[len(x_solution)-1]*1.0936
print(distance)

# Plot the trajectory
# plt.figure(figsize=(8, 6))
# plt.plot(x_solution, y_solution, label='Trajectory')
# plt.title('Golf Ball Trajectory')
# plt.xlabel('X-coordinate (m)')
# plt.ylabel('Y-coordinate (m)')
# plt.legend()
# plt.grid(True)
# plt.show()
