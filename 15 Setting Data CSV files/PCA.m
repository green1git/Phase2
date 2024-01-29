% Read output.csv 
M = readmatrix("output.csv");

% Label the 15 test conditions
label = 1:15;

%% Perform PCA
[V, Y] = pca(M); % Basis, coords

% Plot the data against the two principle components with the highest
% varience
figure;
Y = Y(:, 1:2); % Keep two principle components with the highest varience
scatter(Y(:, 1), Y(:, 2), 45, label, "filled");
%axis equal;
colorbar;
colormap turbo;

%% Perform DBSCAN
%{
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
%}

%% Distance between points

distance = NaN * ones(15);

for i=1:15
    for j = (1+i):15
        %distance(i, j) = 1
        distance(i, j) = sqrt((Y(i, 1) - Y(j, 1))^2 + (Y(i, 2) - Y(j, 2))^2);
    end
end

% Sort from smallest to largest element, and extract the indices in
% original matrix
[min, idx] = sort(distance(:)) 
idx = idx(1:(15*14)/2) % Keep elements not NaN
[rowidx, colidx] = ind2sub(size(distance), idx);


%% Plot

vals = zeros((15*14)/2, 15);

ypoints = zeros(1, (15*14)/2);

for i=1:(15*14)/2
    if i>1
        vals(i, :) = vals(i-1, :);
    end
    vals(i, rowidx(i)) = vals(i, rowidx(i)) + 1;
    vals(i, colidx(i)) = vals(i, colidx(i)) + 1;
    
    max = 0;
    idx = 0;
    for j = 1:15
        if vals(i, j)>max
            max = vals(i, j);
            idx = j;
        end
    end

    ypoints(i) = j;

end

%{
figure; hold on;
for i=1:15
    scatter(1:(15*14)/2,vals(:, i));
    legend;
end
%xlimits = xlim;
%xticks(xlimits(1):0.25:xlimits(2));
%}


figure;
plot(1:(15*14)/2, ypoints);
