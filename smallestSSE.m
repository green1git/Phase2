
new_constants = optimized_results(:,1);

referenceMatrix = [0.0739384020303626, 0.0488096899443226, -0.0250188202889957, 1.05821807886197, 0.179331060066700, -0.0312309275775539, 0.202396726182479, 0.0134972586379540, -0.00119485032521580, 2.78863843817876, 1.20946080088219];
minSSE = Inf; % Initialize minimum SSE
closestMatrixIndex = 0; % To store the index of the closest matrix

for i = 1:length(new_constants)
    currentMatrix = new_constants{i};
    % Calculate sum squared error with the reference matrix
    sse = sum((currentMatrix - referenceMatrix).^2);
    
    if sse < minSSE
        minSSE = sse;
        closestMatrixIndex = i;
    end
end

fprintf('The closest matrix is at index %d with an SSE of %f\n', closestMatrixIndex, minSSE);
