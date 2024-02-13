#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:04:37 2024

@author: archie
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Set the seed
torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)
    
df = pd.read_csv('/Users/archie/Downloads/all_tests.csv')

# Drop the redundant 'Ball Name' column from df_15
df = df.drop(columns=['Ball Name','Date of Test','Ball number',  'Ball Size (inches)'])

Y = df['Setting ID']

X = df.iloc[:, 1:]

# Assuming X is your features and Y is 'Setting ID'
data = pd.concat([X, Y], axis=1)

# Empty lists to hold train and test splits
train_data = []
test_data = []

# Iterate over each group and split
for _, group_data in data.groupby('Setting ID'):
    test_group, train_group = train_test_split(group_data, test_size=len(group_data)-3, random_state=42)
    train_data.append(train_group)
    test_data.append(test_group)

# Concatenate all groups to form the full train and test sets
train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# Separate X and Y again
X_train, X_test = train_data.drop('Setting ID', axis=1), test_data.drop('Setting ID', axis=1)
Y_train, Y_test = train_data['Setting ID'], test_data['Setting ID']

# Separate X and Y again
X_train, X_test = train_data.drop('Setting ID', axis=1), test_data.drop('Setting ID', axis=1)
Y_train, Y_test = train_data['Setting ID'], test_data['Setting ID']

# Scale only the features (X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use transform here, not fit_transform

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)  # If needed for later use

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(X.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Compressed representation in 2D
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, X.shape[1])  # Output layer size same as input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)

# Instantiate the model
model = Autoencoder()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Training the model
num_epochs = 100  # You can adjust this
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    loss = criterion(predicted, X_test_tensor)
    print('Test Loss:', loss.item())


# Extract 2D representations
model.eval()
with torch.no_grad():
    encoded = model.encode(X_test_tensor).numpy()

plt.figure(figsize=(8, 6))
plt.scatter(encoded[:, 0], encoded[:, 1], c=Y_test_tensor)  # Assuming 'encoded' and 'Y_test_tensor' are defined
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Add a colorbar to show the mapping of labels to colors and set its label
cbar = plt.colorbar()
cbar.set_label('Setting ID')

plt.savefig('encoded_groupings.png')
plt.show()


# Convert encoded representations and Setting IDs into a DataFrame for easier manipulation
encoded_df = pd.DataFrame(encoded, columns=['Dim1', 'Dim2'])
encoded_df['Setting ID'] = Y_test_tensor.numpy()  # Make sure Y_test_tensor is converted back to numpy array if it's a tensor

# Calculate centroids for each Setting ID
centroids = encoded_df.groupby('Setting ID').mean().reset_index()

# Extract just the centroid coordinates for distance calculation
centroid_coordinates = centroids[['Dim1', 'Dim2']].values

# Compute the symmetric distance matrix
n_centroids = len(centroid_coordinates)
distance_matrix = np.zeros((n_centroids, n_centroids))

for i in range(n_centroids):
    for j in range(i, n_centroids):
        distance = np.linalg.norm(centroid_coordinates[i] - centroid_coordinates[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # Ensure the matrix is symmetric

# Convert distance matrix to a DataFrame for better readability, using Setting IDs as index and columns
distance_matrix_df = pd.DataFrame(distance_matrix, index=centroids['Setting ID'], columns=centroids['Setting ID'])


plt.figure(figsize=(8, 6))

# Define a list of markers
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v', '<', '>', 'p', 'H', 'h', '+', 'x']

# Ensure we have enough markers for the unique Setting IDs
assert len(markers) >= centroids['Setting ID'].nunique(), "Not enough unique markers for each Setting ID."

# Iterate over each Setting ID and plot with a unique marker
for setting_id, group in centroids.groupby('Setting ID'):
    marker = markers[setting_id % len(markers) - 1]  # Select marker based on Setting ID
    plt.scatter(group['Dim1'], group['Dim2'], marker=marker, label=f'Setting ID {setting_id}')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Create a legend off to the side of the graph
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()  # Adjust the layout to make room for the legend

plt.savefig('encoded_centroids.png')

plt.show()



# Suppose distance_matrix_df is your DataFrame containing the distance matrix
X = distance_matrix_df.values

# Dictionary to store rankings for each k
rankings_for_ks = {}

for k in range(1, 6):  # Loop from k=1 to k=5
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='cosine').fit(X)

    # The distances and indices of the k nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    # For k=1, there's only one neighbor, so we don't need to exclude the first column as it is not self-distance.
    # For k>1, we exclude the first column since it's the distance to itself (0)
    if k == 1:
        avg_distances = distances.mean(axis=1)
    else:
        avg_distances = distances[:, 1:].mean(axis=1)
    
    # Convert to Series for easy handling, using setting IDs as index
    avg_distance_series = pd.Series(avg_distances, index=distance_matrix_df.index)

    # Sort settings by average distance to neighbors - ascending implies closer neighbors
    removal_priority = avg_distance_series.sort_values(ascending=True)

    # Convert to list for a rank of setting IDs
    removal_ranking = removal_priority.index.tolist()

    # Store the ranking for the current k
    rankings_for_ks[k] = removal_ranking

    print(f"Settings ranked for removal based on closeness to neighbors for k={k}:\n", removal_ranking)