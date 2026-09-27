function unnormalized_data = unnormalize_data(normalized_data, data_min, data_max)
    % Unnormalize the data back to its original scale
    unnormalized_data = (normalized_data + 1) .* (data_max - data_min) / 2 + data_min;
end