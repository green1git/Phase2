data = readtable("ball11pp_for_sfs.csv")

num_conditions = 10; 
combinations = nchoosek(1:15, num_conditions);

optimized_results = cell(size(combinations, 1), 3); % First column for constants, second for fval, third for index of combinations

% Set the initial guess for the constants (a1,a2,a3,b1,b2,b3,...,d2) Random
initial_guess = ones(1, 11);

% Set the options for fmincon sqp algo fine for this
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

%Infinite bounds decision varibales can take
lb = -Inf * ones(1, 11); % Lower bounds
ub = Inf * ones(1, 11);  % Upper bounds

for i = 1:size(combinations, 1)
    % sselect subset of data based on the current combination
    current_combination = combinations(i, :);
    subset_data = data(current_combination, :);

    % relevant columns for this subset
    R = subset_data.ReynoldsNumber;
    alpha = subset_data.SpinRatio;
    C_l_observed = subset_data.CoefficientOfLift;
    C_d_observed = subset_data.CoefficientOfDrag;

    % optimization with the subset of data
    [optimized_constants, fval] = fmincon(@(constants) residuals_function(constants, R, alpha, C_l_observed, C_d_observed), initial_guess, [], [], [], [], lb, ub, [], options);

    % Store the results
    optimized_results{i, 1} = optimized_constants;
    %optimized_results{i, 2} = fval;
    optimized_results{i, 2} = current_combination
end

% optimized_results contains the optimized constants and objective function values for each combination
