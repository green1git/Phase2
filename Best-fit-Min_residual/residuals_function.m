function residuals = residuals_function(constants, R, alpha, C_l_observed, C_d_observed)
    %Define constants we are looking for 
    a1 = constants(1);
    a2 = constants(2);
    a3 = constants(3);
    b1 = constants(4);
    b2 = constants(5);
    b3 = constants(6);
    c1 = constants(7);
    c2 = constants(8);
    c3 = constants(9);
    d1 = constants(10);
    d2 = constants(11);
    
    %Define the function used to obtain C_l and C_d
    C_l_predicted = (a1 + a2 ./ R.^5 + a3 ./ R.^7) + (b1 + b2 .* log(R) ./ R.^2 + b3 ./ R.^2) .* alpha;
    C_d_predicted = (c1 + c2 ./ R.^3 + c3 ./ R.^5 + c3 ./ R.^7) + (d1 + d2 .* log(R) ./ R.^2) .* alpha.^2;
    
    %Find the residuals of both functions given inputs 
    residuals = sum((C_l_observed - C_l_predicted).^2) + sum((C_d_observed - C_d_predicted).^2);
end


