%%
% Read output.csv 
M = readmatrix("AllData.csv");
M = readmatrix("../27 Setting Data/AllData.csv");

% Extract features X and labels y
X = M(:, 1:4);
y = M(:, 5);

%% Perform PCA
[V, Y, eigenvalues, tsquared, explained] = pca(X); % Basis, coords

% Plot the data against the two principle components with the highest
% varience
figure;
Y = Y(:, 1:2); % Keep two principle components with the highest varience
scatter(Y(:, 1), Y(:, 2), 45, y, "filled");
axis equal;
colorbar;
colormap turbo;