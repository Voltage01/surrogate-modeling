function Y_pred = predict_VVRR_model_v7(MODEL,X_TEST,D_OUT_TEST)

% predict_VVRR_model_v7 evaluates a trained compressed VV-KRR model on a
% set of test inputs.
%
% INPUTS:
%   MODEL      : trained VV-KRR model structure produced by
%                train_VVRR_CV_learn_v18
%   X_TEST     : matrix of test inputs of size
%                (num. test samples x num. input parameters)
%   D_OUT_TEST : output coordinate vector used when an analytic output
%                kernel is required at prediction time; it can be left
%                empty when the same output grid used in training is adopted
%
% OUTPUT:
%   Y_pred     : predicted outputs in the original (unnormalized) scale,
%                of size (num. test samples x num. output components)
%
% NOTES:
%   - The prediction is performed through the compressed representation
%     stored in MODEL.
%   - If the output kernel is data-driven ('GIVEN'), the stored output
%     kernel matrix is used directly.
%   - If an analytic output kernel is used, the corresponding test-output
%     kernel matrix is built from D_OUT_TEST.

sigma_f = 1;

%% Build input kernel between test samples and training samples
switch MODEL.kx_kernel.type
    case 'RBF'
        Kx_kernel_TEST = kernel_sqexp(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    case 'ardRBF'
        Kx_kernel_TEST = kernel_ardsqexp(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    case 'Matern52'
        Kx_kernel_TEST = kernel_matern52(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    case 'ardMatern52'
        Kx_kernel_TEST = kernel_ardmatern52(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    case 'Matern12'
        Kx_kernel_TEST = kernel_matern12(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    case 'ardMatern12'
        Kx_kernel_TEST = kernel_ardmatern12(X_TEST,MODEL.X,sigma_f,MODEL.params.kx_sigma_l);

    otherwise
        error('INPUT KERNEL NOT DEFINED');
end

%% Build output kernel factor used at prediction time
if strcmp(MODEL.ko_kernel.type, 'GIVEN')

    % Data-driven output kernel: use the stored matrix directly
    B_TEST = MODEL.ko_kernel.B;

    %B_TEST = (1-MODEL.params.ko_sigma_l)*MODEL.ko_kernel.B+eye(size(MODEL.ko_kernel.B))*MODEL.params.ko_sigma_l;

else

    % Analytic output kernel: if no test output grid is provided,
    % reuse the training output grid
    if isempty(D_OUT_TEST)
        D_OUT_TEST = MODEL.D_OUT_ED;
    end

    D_OUT_ED = MODEL.D_OUT_ED;

    switch MODEL.ko_kernel.type

        case 'RBF'
            B_TEST = kernel_sqexp(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);

        case 'Matern52'
            B_TEST = kernel_matern52(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);

        case 'Matern12'
            B_TEST = kernel_matern12(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);

        otherwise
            error('OUTPUT KERNEL NOT DEFINED');
    end

end

%% Evaluate compressed VV-KRR predictor in normalized space
% Y_pred_norm = (Kx * U) * c_tilde * (T' * B_TEST)
Y_pred_norm = (Kx_kernel_TEST * MODEL.kx_kernel.U) * MODEL.coeff * (MODEL.ko_kernel.T' * B_TEST);

%% Undo output normalization
Y_pred = Y_pred_norm .* MODEL.normalization.sigma + MODEL.normalization.mu;