% Load all the 15 setting  data
data = readtable('../all_tests.csv');

R = data.ReynoldsNumber; % Reynolds number column
alpha = data.SpinRatio; % Spin Ratio column
C_l_observed = data.CoefficientOfLift; % C_l in data
C_d_observed = data.CoefficientOfDrag; % C_d in data

% Set the initial guess for the constants (a1,a2,a3,b1,b2,b3,...,d2) Random
initial_guess = ones(1, 11);

% Set the options for fmincon sqp algo fine for this
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

%Infinite bounds decision varibales can take
lb = -Inf * ones(1, 11); % Lower bounds
ub = Inf * ones(1, 11);  % Upper bounds

%Define objective function
objective = @(constants) residuals_function(constants, R, alpha, C_l_observed, C_d_observed);

%Use fmincon to find the constants that minimises the residuals function.  
[optimized_constants, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);

%Generate a grid for plotting the surface
[Alpha_grid, R_grid] = meshgrid(linspace(min(alpha), max(alpha), 100), linspace(min(R), max(R), 100));

% Compute the surfaces using the optimal constants
C_l_surface = (optimized_constants(1) + optimized_constants(2) ./ R_grid.^5 + optimized_constants(3) ./ R_grid.^7) + ...
              (optimized_constants(4) + optimized_constants(5) .* log(R_grid) ./ R_grid.^2 + optimized_constants(6) ./ R_grid.^2) .* Alpha_grid;
C_d_surface = (optimized_constants(7) + optimized_constants(8) ./ R_grid.^3 + optimized_constants(9) ./ R_grid.^5 + optimized_constants(9) ./ R_grid.^7) + ...
              (optimized_constants(10) + optimized_constants(11) .* log(R_grid) ./ R_grid.^2) .* Alpha_grid.^2;

% For the C_d surface plot
figure;
surf(Alpha_grid, R_grid, C_d_surface, 'EdgeColor', 'none');
hold on;
scatter3(alpha, R, C_d_observed, 50, 'r', 'filled');
hold off;
title('$C_d$ vs Spin Ratio ($\alpha$) and Reynolds Number', 'Interpreter', 'latex');
xlabel('Spin Ratio ($\alpha$)', 'Interpreter', 'latex');
ylabel('Reynolds Number (R)', 'Interpreter', 'latex');
zlabel('$C_d$', 'Interpreter', 'latex');
colorbar;
view(-30, 30);

% For the C_l surface plot
figure;
surf(Alpha_grid, R_grid, C_l_surface, 'EdgeColor', 'none');
hold on;
scatter3(alpha, R, C_l_observed, 50, 'r', 'filled');
hold off;
title('$C_l$ vs Spin Ratio ($\alpha$) and Reynolds Number', 'Interpreter', 'latex');
xlabel('Spin Ratio ($\alpha$)', 'Interpreter', 'latex');
ylabel('Reynolds Number (R)', 'Interpreter', 'latex');
zlabel('$C_l$', 'Interpreter', 'latex');
colorbar;
view(-30, 30);

