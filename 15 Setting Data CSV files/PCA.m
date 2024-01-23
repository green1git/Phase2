% Read output.csv 
M = readmatrix("output.csv");

% Label the 15 test conditions
label = linspace(1, 15, 15);

% Perform PCA
[V, Y] = pca(M); % Basis, coords

% Plot the data against the two principle components with the highest
% varience
figure;
scatter(Y(:, 1), Y(:, 2), 45, label, "filled");

colorbar;
colormap hot;
