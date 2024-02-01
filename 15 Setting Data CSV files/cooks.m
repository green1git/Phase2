% Read output.csv 
M = readmatrix("../27 Setting Data/output.csv"); % 27 setting data
%M= readmatrix("output.csv"); % 15 setting data

%% Considering both cl and cd

X = [M(:, 1:2); M(:, 1:2)];
y = [M(:, 3); M(:, 4)];
lm = fitlm(X, y); 
cooksD = lm.Diagnostics.CooksDistance;

% Add cooks distance of cl and cd together for each test condition
cooksD = reshape(cooksD, [size(cooksD, 1)/2, 2]);
cooksD = sum(cooksD, 2);
% Append labels and sort
cooksD(:, 2) = 1:size(cooksD, 1);
removal_order = sortrows(cooksD);
disp(removal_order(:, 2));

%% Taking average first

X = M(:, 1:2);
y = (M(:, 3) + M(:, 4))/2;

lm = fitlm(X, y); 
cooksD = lm.Diagnostics.CooksDistance;

% Append labels and sort
cooksD(:, 2) = 1:size(cooksD, 1);
removal_order = sortrows(cooksD);
disp(removal_order(:, 2));

%% Considering cl and cd seperately

lm_cd = fitlm(M(:, 1:2), M(:, 3)); 
cooksD_cd = lm_cd.Diagnostics.CooksDistance;
% Append labels and sort
cooksD_cd(:, 2) = 1:size(cooksD_cd, 1);
removal_order_cd = sortrows(cooksD_cd);
display(removal_order_cd(:, 2));

lm_cl = fitlm(M(:, 1:2), M(:, 4)); 
cooksD_cl = lm_cl.Diagnostics.CooksDistance;
% Append labels and sort
cooksD_cl(:, 2) = 1:size(cooksD_cl, 1);
removal_order_cl = sortrows(cooksD_cl);
display(removal_order_cl(:, 2));




