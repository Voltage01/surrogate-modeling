function K = kernel_ardsqexp(x,xp,sigma_f,sigma_l)

sigma_l = sigma_l(:)'; % making sure it is a row vector

% anisotropic ellipsoidal squared exponential kernel
h = pdist2(x./sigma_l,xp./sigma_l);
K = sigma_f^2 * exp(-1/2 * h.^2);