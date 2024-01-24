% Read output.csv 
M = readmatrix("output.csv");

% Label the 15 test conditions
label = linspace(1, 15, 15);

%% Perform PCA
[V, Y] = pca(M); % Basis, coords

% Plot the data against the two principle components with the highest
% varience
figure;
Y = Y(:, 1:2); % Keep two principle components with the highest varience
scatter(Y(:, 1), Y(:, 2), 45, label, "filled");
colorbar;
colormap turbo;

%% Perform DBSCAN
clusterDBSCAN.estimateEpsilon(Y,2,15);
clusterDBSCAN.discoverClusters(Y,2,2);

clusterer = clusterDBSCAN('MinNumPoints',2,'Epsilon',0.45447, ...
    'EnableDisambiguation',false);
[idx,cidx] = clusterer(Y);
plot(clusterer,Y,idx)

idx = dbscan(Y,0.403,1);
figure;
gscatter(Y(:,1),Y(:,2), idx, [], [], 45);
mdl = fitlm(Y(:, 1),Y(:, 2));

plotDiagnostics(mdl,'cookd')
