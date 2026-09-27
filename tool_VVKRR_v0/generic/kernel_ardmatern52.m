function K = kernel_ardmatern52(x,xp,sigma_f,sigma_l)

% anisotropic ellipsoidal Matern 5/2 kernel
sigma_l = sigma_l(:)'; % making sure it is a row vector

h = pdist2(x./sigma_l,xp./sigma_l);
K = sigma_f^2 * (1 + sqrt(5)*h + 5/3*h.^2) .* exp(-sqrt(5)*h);