function [normalized_data, data_min, data_max] = normalize_data(data)
    % Normalize each column of the matrix to the range [-1,1]
    data_min = min(data);  % Min value of each column
    data_max = max(data);  % Max value of each column
    normalized_data = 2 * (data - data_min) ./ (data_max - data_min) - 1;
end