% Read output.csv 
M = readmatrix("../27 Setting Data/output.csv"); % 27 setting data
%M= readmatrix("output.csv"); % 15 setting data

% Take transpose, so that each of the 15 test conditions are the features
M = M';

%% Perform PCA
[V, Y, eigenvalues, tsquared, explained] = pca(M); % Basis, coords

%% Compute feature importance

% Initialize a vector to store feature importance
feature_importance = zeros(size(M, 2), 1);

% Loop over each principal component
for i = 1:size(V, 2)
    % The contribution of each feature to this component
    contribution = V(:, i).^2 * explained(i);
    % Accumulate the contribution
    feature_importance = feature_importance + contribution;
end

%% Normalize the feature importance
feature_importance = feature_importance / sum(feature_importance);

%% Append labels and sort

feature_importance(:, 2) = 1:size(feature_importance, 1);
removal_order = sortrows(feature_importance);
disp(removal_order(:, 2));
