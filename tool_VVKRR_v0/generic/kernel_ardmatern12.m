function K = kernel_ardmatern12(x, xp, sigma_f, sigma_l)
    % Anisotropic ellipsoidal Matérn 1/2 kernel (Ornstein-Uhlenbeck)
    % Inputs:
    %   x       - Input points (NxD)
    %   xp      - Test points (MxD)
    %   sigma_f - Signal variance
    %   sigma_l - Length-scale (vector of size D)
    % Output:
    %   K       - Covariance matrix (NxM)
    
    sigma_l = sigma_l(:)';  % Ensure length-scale is a row vector
    
    % Compute scaled Euclidean distance
    h = pdist2(x ./ sigma_l, xp ./ sigma_l);
    
    % Matérn 1/2 kernel function
    K = sigma_f^2 * exp(-h);
end