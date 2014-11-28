function [rank_baseline, rank_specificity] = search_vary_numtestimages(dataset, method)

addpath('../io/'); addpath('utils/');

[~, ~, s, ~, ~, ~] = load_search_parameters(dataset);

if strcmpi(dataset, 'clipart')
    s = s(51:end, 51:end); % XXX: Leave out seed images
end

if strcmpi(method, 'groundtruth')
    load('../../data/specificity_alldatasets.mat');
    eval(['y = specificity.' dataset '.B0;']);
    eval(['z = specificity.' dataset '.B1;']);
elseif strcmpi(method, 'predicted_1000fold')
    load(['../../data/predict_search/' dataset '/predicted_specificity_1000fold.mat']);
    y = y_pred(1, :); z = z_pred(1, :);
end

n_images = length(y);
test_size = 100:10:n_images;
rank_baseline = cell(length(test_size));
rank_specificity = cell(length(test_size));

for idx=1:length(test_size)

    fprintf('\nTest size = %d', test_size(idx));
    test_idx = randsample(n_images, test_size(idx));
    s_test = s(test_idx, test_idx);
    
    % BASELINE
    rank_b = baseline_search(s_test);
    rank_baseline{idx} = rank_b;

    % GROUND TRUTH / PREDICTED SPECIFICITY
    rank_s = specificity_search(s_test, y(test_idx), z(test_idx));
    rank_specificity{idx} = rank_s;

    % SAVE RESULTS
    if strcmpi(method, 'groundtruth')
        save(['../../data/predict_search/' dataset '/groundtruth_logistic.mat'], ...
              'rank_specificity', 'rank_baseline', 'test_size');
    elseif strcmpi(method, 'predicted_1000fold')
        save(['../../data/predict_search/' dataset '/predicted_logistic_1000fold.mat'], ...
              'rank_specificity', 'rank_baseline', 'test_size');
    end
end

end
