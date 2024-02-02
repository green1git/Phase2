import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt  # Corrected import statement

# Load the data
df = pd.read_csv('cooksdata.csv')  # Change to your CSV file path

# Select features and target
X = df[['BallSpeed', 'LaunchAngle', 'SpinRate']]  # Independent variables
y = df['ObservedCarry']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict carry distances
y_pred = model.predict(X_test)

# Calculate errors
errors = y_test - y_pred

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Calculate residuals for the training set
residuals_train = y_train - model.predict(X_train)

# Recalculate MSE based on training residuals
mse_train = np.mean(residuals_train**2)

# Calculate leverages
p = X_train.shape[1] + 1  # Number of predictors + 1 for the intercept
X_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
hat_matrix = X_with_intercept.dot(np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))).dot(X_with_intercept.T)
leverages = np.diag(hat_matrix)

# Calculate Cook's distances
cooks_distances = (residuals_train**2 / (p * mse_train)) * (leverages / (1 - leverages)**2)

# Plot Cook's distances
def plot_cooks_distances(cooks_distances):
    n = len(cooks_distances)  # Number of observations
    threshold = 4 / n  # Common rule of thumb for influential point threshold
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(n), cooks_distances, markerfmt=",", use_line_collection=True)
    plt.axhline(y=threshold, linestyle='--', color='r', label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot")
    plt.legend()
    plt.show()

plot_cooks_distances(cooks_distances)

