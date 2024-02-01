% Load the data
data = readtable('all_tests.csv');

% Get unique BallNames
uniqueBallNames = unique(data.BallName);

% Initialize arrays to store the data
ballNames = cell(numel(uniqueBallNames), 1);
optimizedConstantsArray = zeros(numel(uniqueBallNames), 12); % For 12 constants

for i = 1:numel(uniqueBallNames)
    ballName = strtrim(uniqueBallNames{i}); % Remove leading/trailing spaces
    
    % Extract rows with the current BallName
    subsetTable = data(strcmp(data.BallName, ballName), :);
    
    % Reynolds Numbers, Spin Ratio, C_l, and C_d
    R = subsetTable.ReynoldsNumber .* 10^5;
    alpha = subsetTable.SpinRatio;
    C_l_observed = subsetTable.CoefficientOfLift;
    C_d_observed = subsetTable.CoefficientOfDrag;
    
    % Set the initial guess for the constants
    initial_guess = ones(1, 12); % For 12 constants
    
    % Set the options for fmincon sqp algorithm
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
    
    % Infinite bounds decision variables can take
    lb = -Inf * ones(1, 12); % Lower bounds
    ub = Inf * ones(1, 12);  % Upper bounds
    
    % Define objective function
    objective = @(constants) residuals_function(constants, R, alpha, C_l_observed, C_d_observed);
    
    % Use fmincon to find the constants that minimize the residuals function
    [optimizedConstants, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
    
    % Store the current 'BallName' and optimized constants in the arrays
    ballNames{i} = ballName;
    optimizedConstantsArray(i, :) = optimizedConstants;
end

% Create a table with 'BallName' and the optimized constants as columns
optimizedConstantsTable = table(ballNames, optimizedConstantsArray);

% Create a table for BallNames
ballNameTable = table(optimizedConstantsTable.ballNames, 'VariableNames', {'BallName'});

% Create a table for the constants
constantsTable = array2table(optimizedConstantsTable.optimizedConstantsArray);

% Set the variable names for the constantsTable
constantNames = {'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2'};
constantsTable.Properties.VariableNames = constantNames;

% Concatenate ballNameTable and constantsTable to form the expandedTable
expandedTable = [ballNameTable, constantsTable];

% Display the final table
disp(expandedTable);

% Specify the file name for the CSV
filename = 'all_constants.csv';

% Save the expandedTable as a CSV file
writetable(expandedTable, filename);

