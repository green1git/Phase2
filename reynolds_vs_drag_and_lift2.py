import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Function to fit a line
def linear_fit(x, m, b):
    return m * x + b

# List of golf ball names and corresponding colors
ball_colors = {'Ball-11-PH': 'red', 'Ball-12-PH': 'blue', 'Ball-13-PH': 'green', 'Ball-14-PH': 'orange', 'Ball-15-PH': 'purple', 'Ball-16-PH': 'brown'}

# Initialize subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Iterate through each golf ball
for ball_name in ball_colors:
    # Read data from CSV file
    data = pd.read_csv(f'{ball_name}.csv')

    # Plot Reynolds against Coefficient of Drag
    drag_color = ball_colors[ball_name]
    axes[0].scatter(data['Reynolds'], data['Coefficient_of_Drag'], label=ball_name, color=drag_color)
    
    # Fit a line to the data
    popt_drag, _ = curve_fit(linear_fit, data['Reynolds'], data['Coefficient_of_Drag'])
    drag_line = linear_fit(data['Reynolds'], *popt_drag)
    axes[0].plot(data['Reynolds'], drag_line, linestyle='--', color=drag_color, alpha=0.5)

    # Plot Reynolds against Coefficient of Lift
    lift_color = ball_colors[ball_name]
    axes[1].scatter(data['Reynolds'], data['Coefficient_of_Lift'], label=ball_name, color=lift_color)
    
    # Fit a line to the data
    popt_lift, _ = curve_fit(linear_fit, data['Reynolds'], data['Coefficient_of_Lift'])
    lift_line = linear_fit(data['Reynolds'], *popt_lift)
    axes[1].plot(data['Reynolds'], lift_line, linestyle='--', color=lift_color, alpha=0.5)

# Set plot titles and labels
axes[0].set_title('Reynolds vs. Coefficient of Drag')
axes[0].set_xlabel('Reynolds Number')
axes[0].set_ylabel('Coefficient of Drag')
axes[0].legend()

axes[1].set_title('Reynolds vs. Coefficient of Lift')
axes[1].set_xlabel('Reynolds Number')
axes[1].set_ylabel('Coefficient of Lift')
axes[1].legend()

plt.tight_layout()
plt.show()
