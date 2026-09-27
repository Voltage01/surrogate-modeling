%% Clear everything
close all;
clear;
clc;

scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir);
addpath(fullfile(scriptDir, 'LTspice2Matlab'));

%% Monte Carlo / Latin hypercube settings
N_MC = 500;  % Use 5 or 10 for the first pipeline check

V_VS = 5;
R_C1 = 4.7e3;
R_C2 = 4.7e3;
I_S = 1e-3;

% Differential-input test: equal and opposite 0.5 V AC inputs
V_Vin_1_mag = 0.5;
V_Vin_1_phase = 0;
V_Vin_2_mag = 0.5;
V_Vin_2_phase = 180;

delta = 0.1;  % Independent resistor variation: +/-10%

tic

% Each row is one sample; columns are relative changes to RC1 and RC2.
X_MC = (2*lhsdesign(N_MC, 2) - 1) * delta;

figure;
histogram(R_C1 * (1 + X_MC(:,1)));
hold on;
histogram(R_C2 * (1 + X_MC(:,2)));
xlabel('Collector resistance (Ohms)');
ylabel('Count');
legend('R_C1 samples', 'R_C2 samples');

index_NAN = zeros(N_MC, 1);
freq = [];

% These matrices will be initialized after the first successful AC run.
V2_diff_input = [];
V3_diff_input = [];
V2_common_input = [];
V3_common_input = [];

ltspiceExe = '/Applications/LTspice.app/Contents/MacOS/LTspice';
netlistFile = fullfile(scriptDir, 'basic-differential-amplifier.cir');
paramsFile = fullfile(scriptDir, 'DIFFERENTIAL_PARAMS.cir');
rawFile = fullfile(scriptDir, 'basic-differential-amplifier.raw');

%% Run both input modes for every parameter sample
for p1 = 1:N_MC

    PR_C1 = R_C1 * (1 + X_MC(p1,1));
    PR_C2 = R_C2 * (1 + X_MC(p1,2));
    PV_VS = V_VS;
    PI_S = I_S;

    % mode = 1: differential input
    % mode = 2: common-mode input
    for mode = 1:2

        if mode == 1
            PV_AC1 = V_Vin_1_mag;
            PV_PHASE1 = V_Vin_1_phase;
            PV_AC2 = V_Vin_2_mag;
            PV_PHASE2 = V_Vin_2_phase;
        else
            PV_AC1 = 1;
            PV_PHASE1 = 0;
            PV_AC2 = 1;
            PV_PHASE2 = 0;
        end

        % Write this sample's circuit and input parameters.
        fid2 = fopen(paramsFile, 'w');
        if fid2 == -1
            error('Could not open parameter file for writing: %s', paramsFile);
        end

        fprintf(fid2, '* Parameters for differential-amplifier AC simulation\n\n');
        fprintf(fid2, '.PARAM PR_C1=%.15g\n', PR_C1);
        fprintf(fid2, '.PARAM PR_C2=%.15g\n', PR_C2);
        fprintf(fid2, '.PARAM PV_VS=%.15g\n', PV_VS);
        fprintf(fid2, '.PARAM PI_S=%.15g\n', PI_S);
        fprintf(fid2, '.PARAM PV_Vin_1_amp=%.15g\n', PV_AC1);
        fprintf(fid2, '.PARAM PV_Vin_1_phase=%.15g\n', PV_PHASE1);
        fprintf(fid2, '.PARAM PV_Vin_2_amp=%.15g\n', PV_AC2);
        fprintf(fid2, '.PARAM PV_Vin_2_phase=%.15g\n', PV_PHASE2);
        fclose(fid2);

        % Run LTspice. system() waits for the batch command to finish.
        command = sprintf('"%s" -b "%s"', ltspiceExe, netlistFile);
        [ans_sys, cmdout] = system(command);

        if ans_sys ~= 0
            fprintf(2, 'LTspice failed for sample %d, mode %d.\n', p1, mode);
            fprintf(2, '%s\n', cmdout);
            index_NAN(p1) = 1;
            break;
        end

        % Import the AC result.
        sim_res = LTspice2Matlab(rawFile);

        f_raw = reshape(sim_res.freq_vect, 1, []);
        names = sim_res.variable_name_list;

        idxV2 = find(strcmpi(names, 'V(2)'), 1);
        idxV3 = find(strcmpi(names, 'V(3)'), 1);

        if isempty(idxV2) || isempty(idxV3)
            error('V(2) and/or V(3) were not found in the raw-file variable list.');
        end

        v2_raw = sim_res.variable_mat(idxV2, :);
        v3_raw = sim_res.variable_mat(idxV3, :);

        if any(isnan(v2_raw)) || any(isinf(v2_raw)) || ...
           any(isnan(v3_raw)) || any(isinf(v3_raw))
            fprintf(2, 'Invalid AC data for sample %d, mode %d.\n', p1, mode);
            index_NAN(p1) = 1;
            break;
        end

        % Initialize storage using the frequency vector from the first run.
        if isempty(freq)
            freq = f_raw;
            nFreq = numel(freq);

            V2_diff_input = complex(nan(N_MC, nFreq));
            V3_diff_input = complex(nan(N_MC, nFreq));
            V2_common_input = complex(nan(N_MC, nFreq));
            V3_common_input = complex(nan(N_MC, nFreq));

        elseif ~isequal(freq, f_raw)
            error('Frequency grid changed between simulations.');
        end

        % Save the output from this mode for this sample.
        if mode == 1
            V2_diff_input(p1,:) = v2_raw;
            V3_diff_input(p1,:) = v3_raw;
        else
            V2_common_input(p1,:) = v2_raw;
            V3_common_input(p1,:) = v3_raw;
        end
    end
end

MC_training = toc;

%% Remove samples where either AC run failed
if isempty(freq)
    error('No simulations completed successfully; there is no AC data to save.');
end

good = (index_NAN == 0);

X_MC = X_MC(good,:);
V2_diff_input = V2_diff_input(good,:);
V3_diff_input = V3_diff_input(good,:);
V2_common_input = V2_common_input(good,:);
V3_common_input = V3_common_input(good,:);

N_MC = size(X_MC, 1);

%% Calculate gains and CMRR
% Differential input is 0.5 - (-0.5) = 1 V.
Vdiff_input = V_Vin_1_mag + V_Vin_2_mag;

% Common-mode input is 1 V on both inputs.
Vcommon_input = 1;

% Use the same single-ended output, node 2, for both gains.
Ad = V2_diff_input ./ Vdiff_input;
Acm = V2_common_input ./ Vcommon_input;

CMRR_dB = 20 * log10(abs(Ad ./ Acm));

% Also retain modal output responses for inspection.
Vdiff_out_diff_input = V2_diff_input - V3_diff_input;
Vcm_out_common_input = (V2_common_input + V3_common_input) / 2;
Vdiff_out_common_input = V2_common_input - V3_common_input;

%% Save the dataset
save('diff_amp_data.mat', ...
    'X_MC', 'freq', 'Ad', 'Acm', 'CMRR_dB', ...
    'V2_diff_input', 'V3_diff_input', ...
    'V2_common_input', 'V3_common_input', ...
    'Vdiff_out_diff_input', 'Vcm_out_common_input', ...
    'Vdiff_out_common_input', 'N_MC', 'MC_training');

%% Plot response curves
figure;
semilogx(freq, 20*log10(abs(Ad.')));
xlabel('Frequency (Hz)');
ylabel('Differential gain at node 2 (dB)');
title('Differential-mode gain');

figure;
semilogx(freq, 20*log10(abs(Acm.')));
xlabel('Frequency (Hz)');
ylabel('Common-mode gain at node 2 (dB)');
title('Common-mode gain');

figure;
semilogx(freq, CMRR_dB.');
xlabel('Frequency (Hz)');
ylabel('CMRR (dB)');
title('CMRR versus frequency');