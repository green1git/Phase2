
#why is there only 12 when theres 18 initial conditions
#this isnt for the real error data neeed to run initiaal conditions through the updating cl and cd model
#only 2 lines influential atm according to influential cooks idstance rule of thumb n/4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('cooksdata.csv')

df['Error_Observed_Simulated'] = df['ObservedCarry'] - df['SIMULATED']

# Grouupedp by initial conditions and calculate average error across all balls for each condition
grouped = df.groupby(['BallSpeed', 'LaunchAngle', 'SpinRate'])
df_avg_error = grouped['Error_Observed_Simulated'].mean().reset_index()


# The predictors are the initial conditions, and the target is the average error
X = df_avg_error[['BallSpeed', 'LaunchAngle', 'SpinRate']]
y_error = df_avg_error['Error_Observed_Simulated']

X_train, X_test, y_train, y_test = train_test_split(X, y_error, test_size=0.2, random_state=42)

#linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict the average errors
y_pred = model.predict(X_test)

# models performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# residuals for the training set
residuals_train = y_train - model.predict(X_train)

# leverages
p = X_train.shape[1] + 1  # the plus one is for the intercept
X_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
hat_matrix = X_with_intercept.dot(np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))).dot(X_with_intercept.T)
leverages = np.diag(hat_matrix)

# cooks distances for the training set
mse_train = np.mean(residuals_train**2)
cooks_distances = (residuals_train**2 / (p * mse_train)) * (leverages / (1 - leverages)**2)

def plot_cooks_distances(cooks_distances):
   
    n = len(cooks_distances)  
    threshold = 4 / n  # influential threshold point
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(n), cooks_distances, markerfmt=",", use_line_collection=True)
    plt.axhline(y=threshold, linestyle='--', color='r', label=f'Threshold = (4/n){threshold:.3f}')
    plt.xlabel('Initial Condition')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot for Average Error by Initial Conditions")
    plt.legend()
    plt.show()

plot_cooks_distances(cooks_distances)

print(df_avg_error.shape[0])

