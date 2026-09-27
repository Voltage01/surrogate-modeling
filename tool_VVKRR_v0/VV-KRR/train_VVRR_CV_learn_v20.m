function [MODEL] = train_VVRR_CV_learn_v20(X,Ygiven,kx_type,ko_type,D_OUT_ED,Nfold,tol,normalized,error_type)

% train_VVRR_CV_learn_v18 trains a Vector-Valued Kernel Ridge Regression
% (VV-KRR) model using the training pairs X and Y and estimates the
% hyperparameters by minimizing a K-fold cross-validation error.
%
% INPUTS:
%   X         : matrix of input training samples of size
%               (num. realizations x num. input parameters)
%   Ygiven    : matrix of output training samples of size
%               (num. realizations x num. output components)
%   kx_type   : string defining the kernel used for the input parameters
%   ko_type   : either
%               - a string defining the output kernel, or
%               - a precomputed output-kernel matrix
%   D_OUT_ED  : output coordinate vector used when an analytic output
%               kernel is selected; it can be left empty when ko_type
%               is a precomputed matrix
%   Nfold     : number of folds used in cross-validation
%   tol       : tolerance used for the compressed eigendecomposition
%   normalized: output normalization mode
%               'o' = no normalization
%               's' = z-score normalization
%   error_type: error metric used in cross-validation
%               'L2' or 'L1'
%
% OUTPUT:
%   MODEL     : structure collecting the trained VV-KRR model, including
%               kernel information, normalization parameters, compressed
%               coefficients, retained eigenspaces, and training cost
%
% NOTES:
%   - If ko_type is a numeric matrix, it is interpreted as a fixed
%     data-driven output kernel.
%   - If ko_type is a string, the corresponding analytic output kernel
%     is built and its hyperparameters are optimized together with those
%     of the input kernel.
%   - The final model is trained on the full dataset after the
%     cross-validation-based hyperparameter estimation.

%% Initialization

disp('********************************************')
disp('Model Initialization')

% Start timer for the whole training procedure
tStart = tic;

%% Sizes
[N_ED,N_param] = size(X);
N_OUT = size(Ygiven,2);

%% 1) Normalize outputs
switch lower(normalized)

    % Original outputs
    case 'o'
         Y = Ygiven;
         mu_only = zeros(1,N_OUT);
         s_only = ones(1,N_OUT);

    % Standardization
    case 's'
        [Y,mu_only,s_only] = normalize(Ygiven, 'zscore');
        index_NAN = find(isnan(Y));
        Y(index_NAN) = 0;

    otherwise
        error('NORMALIZATION NOT DEFINED');
end

%% Output kernel
if isnumeric(ko_type) && ismatrix(ko_type)

    % Precomputed data-driven output kernel
    [T,Mu] = eig_compr_psd_v2(ko_type,tol);

    D_OUT_ED = [];
    N_param_ko = 0;

else

    % Analytic output kernel
    T = [];
    Mu = [];

    if isempty(D_OUT_ED)
        D_OUT_ED = linspace(-1,1,N_OUT).';
    end
    
    switch ko_type
        case 'RBF' 
            N_param_ko = 1;
            
        case 'Matern52'                   
            N_param_ko = 1;

        case 'Matern12'             
            N_param_ko = 1;
        
        otherwise
            error('OUTPUT KERNEL NOT DEFINED');
    end
end

%% Input kernel: define number of hyperparameters
switch kx_type
    case 'RBF' 
        N_param_kx = 1;
                        
    case 'ardRBF'
        N_param_kx = N_param;
            
    case 'Matern52'                   
        N_param_kx = 1;

    case 'Matern12'               
        N_param_kx = 1;
                       
    case 'ardMatern52'
        N_param_kx = N_param;
                        
    case 'ardMatern12'
        N_param_kx = N_param;
                        
    otherwise
        error('INPUT KERNEL NOT DEFINED');
end

%% Initial guess for hyperparameters (in log10 scale)
% changed
sigma_l_ko_log_0 = log10(0.001 * ones(N_param_ko, 1));
sigma_l_kx_log_0 = log10(10 * ones(N_param_kx, 1));
eta_log_0 = -4;

hyperparam_log_0 = [sigma_l_ko_log_0; sigma_l_kx_log_0; eta_log_0];

%% Optimizer setup
opts = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'MaxIter', 300, ...
    'MaxFunctionEvaluations', 3000, ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', 1e-2);

disp('********************************************')
disp('Hyperparameters estimation')

rng(1,'twister');
% Training-sample partitioning via K-fold cross-validation
c_part = cvpartition(N_ED,'Kfold',Nfold);

% Define the cross-validation objective function
ObjFcn = @(hyperparam_log) EigDec_Discrete_Sylvestre_CV_fmin_v18( ...
    X,Y,kx_type,ko_type,D_OUT_ED,T,Mu,c_part,mu_only,s_only,tol,error_type,hyperparam_log);

% Optimize hyperparameters
[hyperparam_log_opt, ~] = fminunc(ObjFcn, hyperparam_log_0, opts);
  
% Extract optimal hyperparameters
sigma_l_ko_opt = hyperparam_log_opt(1:N_param_ko);
sigma_l_kx_opt = hyperparam_log_opt(N_param_ko + (1:N_param_kx));
eta_opt = hyperparam_log_opt(N_param_ko + N_param_kx + 1);

disp('********************************************')
disp('Optimal Hyperparameters')

%% Final model training
disp('********************************************')
disp('Model Training')

% Define model structure
MODEL.X = X;

MODEL.kx_kernel.type = kx_type;
MODEL.kx_kernel.U = [];
MODEL.kx_kernel.Lambda = [];

MODEL.params.eta = 10.^(eta_opt);

if isnumeric(ko_type) && ismatrix(ko_type)
    MODEL.ko_kernel.type = 'GIVEN';
    MODEL.ko_kernel.B = ko_type;
    MODEL.ko_kernel.T = T;
    MODEL.ko_kernel.Mu = Mu;

    MODEL.params.ko_sigma_l = [];
    MODEL.params.kx_sigma_l = 10.^(sigma_l_kx_opt);
    MODEL.D_OUT_ED = [];
    
else
    MODEL.ko_kernel.type = ko_type;
    MODEL.ko_kernel.B = [];

    MODEL.params.ko_sigma_l = 10.^(sigma_l_ko_opt);
    MODEL.params.kx_sigma_l = 10.^(sigma_l_kx_opt);
    MODEL.D_OUT_ED = D_OUT_ED;
    
end

MODEL.tol = tol;
MODEL.normalization.type = normalized;
MODEL.normalization.mu = mu_only;
MODEL.normalization.sigma = s_only;
MODEL.coeff = [];

% Build the final model on the whole dataset
MODEL = build_VVRR_model_v10(MODEL,Y,tol);

% Store total training cost
model_training_cost = toc(tStart);
MODEL.training_cost = model_training_cost;

disp('********************************************')
disp('Model TRAINED!!!!!!')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ObjFcn] = EigDec_Discrete_Sylvestre_CV_fmin_v18(X_ED_norm,Y_ED,kx_type,ko_type,D_OUT_ED,T,Mu,c_part,mu_only,s_only,tol,error_type,hyperparam_log)

% Evaluate the K-fold cross-validation objective associated with the
% current hyperparameter vector.

% Number of folds
Kfold = c_part.NumTestSets;
ObjFcn = 0;

N_param_x = size(X_ED_norm,2);
eta = 10.^(hyperparam_log(end));

%% Output kernel
if isnumeric(ko_type) && ismatrix(ko_type)
        
    N_param_ko = 0; 
    B_full = ko_type;
            
else
    switch ko_type
        case 'RBF'
            N_param_ko = 1;
            sigma_f = 1;

            sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
            sigma_ko_l = sigma_ko_l(:);
                    
            B_full = kernel_sqexp(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);
                        
        case 'Matern52'
            N_param_ko = 1; 
            sigma_f = 1;

            sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
            sigma_ko_l = sigma_ko_l(:);

            B_full = kernel_matern52(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);
                                                

        case 'Matern12'
            N_param_ko = 1; 
            sigma_f = 1;

            sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
            sigma_ko_l = sigma_ko_l(:);

            B_full = kernel_matern12(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);

        otherwise
            error('OUTPUT KERNEL NOT DEFINED');
    end
                                
    % Compute compressed eigendecomposition of the output kernel
    [T,Mu] = eig_compr_psd_v2(B_full,tol);
end

%% Cross-validation loop
for KK = 1:Kfold
    % Training set for the current fold
    X_ED_K_norm = X_ED_norm(c_part.training(KK),:);
    Y_ED_K = Y_ED(c_part.training(KK),:);

    % Validation set for the current fold
    X_VAL_K_norm = X_ED_norm(c_part.test(KK),:);
    Y_VAL_K = Y_ED(c_part.test(KK),:);

    % Build input kernel
    switch kx_type
        case 'RBF'
            N_param_kx = 1;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);

            Kx = kernel_sqexp(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_sqexp(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                
        case 'ardRBF'
            N_param_kx = N_param_x;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);

            Kx = kernel_ardsqexp(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_ardsqexp(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

        case 'Matern52'
            N_param_kx = 1;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);                    
                    
            Kx = kernel_matern52(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_matern52(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                
        case 'ardMatern52'
            N_param_kx = N_param_x;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);

            Kx = kernel_ardmatern52(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_ardmatern52(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

        case 'Matern12'
            N_param_kx = 1;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);

            Kx = kernel_matern12(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_matern12(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
       
        case 'ardMatern12'
            N_param_kx = N_param_x;
            sigma_f = 1;                    

            sigma_kx_l  = 10.^(hyperparam_log(N_param_ko + (1:N_param_kx)));
            sigma_kx_l = sigma_kx_l(:);

            Kx = kernel_ardmatern12(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
            Kx_VAL = kernel_ardmatern12(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

        otherwise
            error('INPUT KERNEL NOT DEFINED');
    end

    % Compressed eigendecomposition of the input kernel
    [U,Lambda] = eig_compr_psd_v2(0.5*(Kx+Kx'),tol);

    % Transform training outputs
    Y_ED_tilde = U' * Y_ED_K * T;

    if isempty(Lambda) || isempty(Mu)                
        ObjFcn = Inf;
        return;
    else
        lambda_reg = eta * mean(diag(Lambda)) * mean(diag(Mu));%Lambda(end,end) * Mu(end,end);
    end
            
    % Solve the compressed Sylvester equation
    c_tilde = Y_ED_tilde ./ (diag(Lambda) .* diag(Mu)' + lambda_reg);
    
    % Validation prediction
    Y_VAL_MOKK = (Kx_VAL * U) * c_tilde * (T' * B_full);
      
    % Undo normalization
    Y_VAL_MOKK_UN = Y_VAL_MOKK .* s_only + mu_only;
    Y_VAL_K_UN = Y_VAL_K .* s_only + mu_only;

    % Accumulate fold error
    eps_norm = 1e-9;
            
    switch error_type
        case 'L2'
            ObjFcn = ObjFcn + norm(Y_VAL_K_UN(:) - Y_VAL_MOKK_UN(:), 'fro').^2 / ...
                              (norm(Y_VAL_K_UN(:), 'fro') + eps_norm).^2;
        
        case 'L1'
            ObjFcn = ObjFcn + sum(abs(Y_VAL_K_UN(:) - Y_VAL_MOKK_UN(:))) / ...
                              (sum(abs(Y_VAL_K_UN(:))) + eps_norm);
        
        otherwise               
            error('ERROR TYPE NOT DEFINED');
    end
end

% Average cross-validation error
ObjFcn = ObjFcn / Kfold;   
            
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MODEL] = build_VVRR_model_v10(MODEL,Y,tol)

% Build the final VV-KRR model on the full training set using the
% optimized hyperparameters stored in MODEL.

% Read training inputs
X = MODEL.X;
sigma_f = 1;

%% Output kernel
if strcmp(MODEL.ko_kernel.type, 'GIVEN')

    T = MODEL.ko_kernel.T;
    Mu = MODEL.ko_kernel.Mu;
    B_full = MODEL.ko_kernel.B;

else

    D_OUT_ED = MODEL.D_OUT_ED;

    switch MODEL.ko_kernel.type
        case 'RBF'
            B_full = kernel_sqexp(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);
                    
        case 'Matern52'
            B_full = kernel_matern52(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);
                                        
        case 'Matern12'
            B_full = kernel_matern12(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);

        otherwise
            error('OUTPUT KERNEL NOT DEFINED');
    end
    
    % Compressed eigendecomposition of the output kernel
    [T,Mu] = eig_compr_psd_v2(B_full,tol);
end

%% Input kernel
switch MODEL.kx_kernel.type
    case 'RBF'
        K_kernel = kernel_sqexp(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    case 'ardRBF'
        K_kernel = kernel_ardsqexp(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    case 'Matern52'
        K_kernel = kernel_matern52(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    case 'ardMatern52'
        K_kernel = kernel_ardmatern52(X,X,sigma_f,MODEL.params.kx_sigma_l);

    case 'Matern12'
        K_kernel = kernel_matern12(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    case 'ardMatern12'
        K_kernel = kernel_ardmatern12(X,X,sigma_f,MODEL.params.kx_sigma_l);

    otherwise
        error('INPUT KERNEL NOT DEFINED');
end
    
% Compressed eigendecomposition of the input kernel
[U,Lambda] = eig_compr_psd_v2(0.5*(K_kernel+K_kernel'),tol);

%% Transform outputs
Y_tilde = U' * Y * T;

%% Compute compressed coefficients
lambda_reg = MODEL.params.eta * mean(diag(Lambda)) * mean(diag(Mu));%Lambda(end,end) * Mu(end,end);
c_tilde = Y_tilde ./ (diag(Lambda) .* diag(Mu)' + lambda_reg);

%% Store model data
MODEL.coeff = c_tilde;
MODEL.ko_kernel.T = T;
MODEL.ko_kernel.Mu = Mu;
MODEL.kx_kernel.U = U;
MODEL.kx_kernel.Lambda = Lambda;

MODEL.nx = size(U,2);
MODEL.no = size(T,2);

end