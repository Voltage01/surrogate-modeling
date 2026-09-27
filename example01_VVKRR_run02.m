% Reproducibility script for Example 1 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the VV-KRR results for Example 1, including:
% 1) the VV-KRR entry corresponding to model #4 in Table 3,
% 2) the scatter plot in Fig. 3,
% 3) the parametric plot in Fig. 4.
%
% By changing the input kernel kx and the output kernel ko,
% it is also possible to reproduce the other VV-KRR configurations
% reported in Table 3 of the paper.

clear all;
clc;
close all;

scriptDir = fileparts(mfilename('fullpath'));

% Add the VV-KRR toolbox folder and all its subfolders.
% The folder "tool_VVKRR_v0" is assumed to be located in the same
% directory as this main script.
addpath(genpath(fullfile(scriptDir, '..', '..', 'tool_VVKRR_v0')));


% Plot/font settings used in the figures
FS = 15;
FN = 'Times';
FW = 'normal';
FA = 'normal';


%% Load dataset for Example 1
% The file Example01_dataset.mat contains:
%   X_ED_norm   : normalized training input samples;
%   Y_ED        : training output responses;
%   X_TEST_norm : normalized test input samples;
%   Y_TEST      : test output responses;
%   freq        : frequency vector.

load(fullfile(scriptDir, 'diff_amp_data.mat'))

%% MC plot
figure
plot(freq,Y_TEST,'k')

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('Magnitude (dB)', 'FontSize', FS, 'FontName', FN);

title('MC example 1 computed on the test set')
% Main plot aesthetics
axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');

%% Modeling

% Compression / truncation tolerance used in the VV-KRR training
tol = 1e-6;

% Build the output kernel matrix from the empirical correlation
% of the training outputs
B = corr(Y_ED);

% Replace NaN entries, which may appear if some output components
% have zero variance in the training set
index_B = find(isnan(B));
B(index_B) = 0;

% Enforce symmetry of the output kernel matrix
B = 0.5*(B+B.');

% Start training timer
t1 = tic;

% Train the VV-KRR model.
%
% X_ED_norm : normalized training inputs
% Y_ED      : training outputs
% kx        : input kernel type
% ko        : output kernel matrix or analytic output kernel type
% []        : optional extra arguments not used here
% 5         : number of CV folds
% tol       : compression tolerance
%
% By changing kx and ko, it is possible to reproduce the VV-KRR
% results reported in Table 3 of the paper.
kx = 'ardMatern52';   % or 'Matern52'; 'RBF'; 'ardRBF'; 'Matern12'; 'ardMatern12'
ko = B;               % or 'Matern52'; 'RBF'; 'Matern12'

MODEL = train_VVRR_CV_learn_v20(X_ED_norm,Y_ED,kx,ko,[],5,tol,'s','L2');


% Stop training timer
time_training = toc(t1);

% Start prediction timer
t2 = tic;

% Predict outputs on the test set
Y_pred = predict_VVRR_model_v7(MODEL,X_TEST_norm,[]);

% Stop prediction timer
time_prediction = toc(t2);

%% Compute error metrics

% Small constant to avoid division by zero in relative errors
eps_norm = 1e-12;

% Relative L2 error over the full output matrix, expressed in percent
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / (norm(Y_TEST, 'fro') + eps_norm) * 100;

% Relative Linf error over all output entries, expressed in percent
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)), inf) / norm(abs(Y_TEST(:)), inf) * 100;

%% Display summary in Command Window
fprintf('\n\n---------------------\n');
fprintf('--- VV-KRR MODEL SUMMARY ---\n');

% Number of training samples
fprintf('Number of training samples: %d\n', size(X_ED_norm,1));

% Input and output kernel types actually stored in the trained model
fprintf('Input kernel kx: %s\n', MODEL.kx_kernel.type);
fprintf('Output kernel ko: %s\n', MODEL.ko_kernel.type);

% Compression tolerance
fprintf('Tolerance: %.1e\n', MODEL.tol);

% Compression summary
fprintf('Retained input modes nx: %d\n', MODEL.nx);
fprintf('Retained output modes no: %d\n', MODEL.no);
fprintf('Number of retained coefficients: %d\n', MODEL.nx * MODEL.no);

% Timing information
fprintf('Training time: %.4f seconds\n', time_training);
fprintf('Prediction time: %.4f seconds\n', time_prediction);

% Error metrics
fprintf('L2 relative error: %.1f\n', L2_rel_Error);
fprintf('Linf relative error: %.1f\n', Linf_rel_Error);


%% Scatter plot
figure;
hold on;
set(gca, 'FontSize', FS, 'FontName', FN, 'FontWeight', FW, 'FontAngle', FA);

% Scatter plot for the current VV-KRR model
scatter(Y_TEST(:), Y_pred(:), 30, 'm', 'x', 'LineWidth', 1.2);

% Reference line y = x, corresponding to perfect prediction
min_val = min([Y_TEST(:); Y_pred(:)]);
max_val = max([Y_TEST(:); Y_pred(:)]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.2);

% Axis labels and formatting
xlabel('Actual Values', 'FontSize', FS, 'FontName', FN);
ylabel('Model Prediction', 'FontSize', FS, 'FontName', FN);
legend('Model #4', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial');

% Styling
axis square;
axis tight;
grid on;
box on;
set(gca, 'LineWidth', 1.2);
set(gcf, 'Color', 'w');

%% Parametric comparison between reference and predicted responses
figure;
hold on;
set(gca, 'FontSize', FS, 'LineWidth', 1.2, 'FontName', FN, 'FontWeight', FW, 'FontAngle', FA);

% Plot reference and model prediction for selected test examples
plot(freq/1e9, Y_TEST(1,:), 'k-', 'LineWidth', 2.5);
plot(freq/1e9, Y_pred(1,:), 'm--', 'LineWidth', 2);

% Plot additional test examples sampled every 4 rows up to row 20
plot(freq/1e9, Y_TEST(1:5:25,:), 'k-', 'LineWidth', 2.5);
plot(freq/1e9, Y_pred(1:5:25,:), 'm--', 'LineWidth', 2);

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('Magnitude (dB)', 'FontSize', FS, 'FontName', FN);

% Legend
legend({'Reference', 'Model #4'}, 'FontSize', FS, 'Location', 'best', 'FontName', 'Arial');

% Main plot aesthetics
axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');

%% Inset plot: output kernel / correlation matrix
D_OUT_TEST = [];

if strcmp(MODEL.ko_kernel.type, 'GIVEN')
    B_TEST = MODEL.ko_kernel.B;
else
    if isempty(D_OUT_TEST)
        D_OUT_TEST = MODEL.D_OUT_ED;
    end

    D_OUT_ED = MODEL.D_OUT_ED;

    switch MODEL.ko_kernel.type
        case 'RBF'
            sigma_f = 1;
            B_TEST = kernel_sqexp(D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        case 'Matern52'
            sigma_f = 1;
            B_TEST = kernel_matern52(D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        case 'Matern12'
            sigma_f = 1;
            B_TEST = kernel_matern12(D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        otherwise
            error('Unsupported output kernel type for inset plot: %s', MODEL.ko_kernel.type);
    end
end

% Create a small inset showing the output-kernel / correlation matrix
inset_pos = [0.15, 0.27, 0.30, 0.30];  % [x, y, width, height] in normalized units
inset_axes = axes('Position', inset_pos);

% Display the matrix associated with the output correlation structure
imagesc(freq/1e9, freq/1e9, B_TEST);
set(inset_axes, 'FontSize', FS-2, 'LineWidth', 1, 'FontName', FN);
colormap(inset_axes, parula);
axis square tight;
box on;

xlabel('', 'FontSize', FS-2);
ylabel('', 'FontSize', FS-2);
title('B Matrix', 'FontSize', FS-2, 'FontName', FN);

% Turn off x/y ticks in the inset to keep it clean
set(inset_axes, 'XTick', [], 'YTick', []);

% Add a small colorbar for the inset
colorbar('Position', [0.43, 0.27, 0.01, 0.30]);