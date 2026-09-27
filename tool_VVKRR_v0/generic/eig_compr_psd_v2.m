function [V, D] = eig_compr_psd_v2(A, tol)
%EIG_COMPR_PSD_V2
% PSD eigendecomposition with truncation based on discarded energy.
%
% Inputs:
%   A   : symmetric / PSD matrix
%   tol : allowed discarded relative energy
%         e.g.
%           tol = 1e-2  -> keep 99% energy
%           tol = 1e-3  -> keep 99.9% energy
%
% Outputs:
%   V   : retained eigenvectors
%   D   : retained eigenvalues on the diagonal
%
% Criterion:
%   sum(discarded eigenvalues) / sum(all eigenvalues) <= tol

if tol < 0 || tol >= 1
    error('tol must satisfy 0 <= tol < 1. Example: tol = 1e-3');
end

% enforce symmetry
A = 0.5 * (A + A.');

% eigendecomposition
[Vfull, lam] = eig(A, 'vector');
lam = real(lam);

% sort descending
[lam, idx] = sort(lam, 'descend');
Vfull = Vfull(:, idx);

% clip negative eigenvalues (numerical noise)
lam(lam < 0) = 0;

% remove exact / tiny modes
if all(lam == 0)
    V = zeros(size(A,1), 0);
    D = zeros(0,0);
    return;
end

keep = lam > max(lam) * 1e-14;
lam = lam(keep);
Vfull = Vfull(:, keep);

% cumulative retained energy
cumE = cumsum(lam) / sum(lam);

% keep enough modes so that discarded energy <= tol
target = 1 - tol;
r = find(cumE >= target, 1, 'first');

if isempty(r)
    r = numel(lam);
end

V = Vfull(:, 1:r);
D = diag(lam(1:r));

end