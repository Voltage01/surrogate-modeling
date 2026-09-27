function K = kernel_sqexp(x,xp,sigma_f,sigma_l)

% isotropic ellipsoidal squared exponential kernel
K = sigma_f^2 * exp(-1/2 * pdist2(x,xp).^2/sigma_l^2);