function K = kernel_matern12(x, xp, sigma_f, sigma_l)
    % Matérn 1/2 kernel (Ornstein-Uhlenbeck)
    % Inputs:
    %   x       - Input points (NxD)
    %   xp      - Test points (MxD)
    %   sigma_f - Signal variance
    %   sigma_l - Scalar length-scale
    % Output:
    %   K       - Covariance matrix (NxM)
    
    % Compute Euclidean distance
    h = pdist2(x, xp, 'euclidean') / sigma_l;
    
    % Matérn 1/2 kernel function
    K = sigma_f^2 * exp(-h);
end