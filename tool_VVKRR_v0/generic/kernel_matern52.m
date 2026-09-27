function K = kernel_matern52(x,xp,sigma_f,sigma_l)

% isotropic ellipsoidal Matern 5/2 kernel
r = pdist2(x,xp);
K = sigma_f^2 * (1 + sqrt(5)*r/sigma_l + 5*r.^2/(3*sigma_l^2)) .* exp(-sqrt(5)*r/sigma_l);