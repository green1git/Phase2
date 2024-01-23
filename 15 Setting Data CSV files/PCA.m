M = readmatrix("output.csv");
label = linspace(1, 15, 15);

[V, Y] = pca(M); %Basis, coords

figure;
scatter(Y(:, 1), Y(:, 2), 45, label, "filled");

colorbar;
colormap hot;
