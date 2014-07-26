function [rank_baseline, rank_specificity] = search_vary_numtestimages(dataset, method)

addpath('../aux_functions/');

[~, ~, s, ~, ~, ~] = load_search_parameters(dataset);

if strcmpi(method, 'groundtruth')
    load('../../data/specificity_alldatasets.mat');
    eval(['y = specificity.' dataset '.B0;']);
    eval(['z = specificity.' dataset '.B1;']);
else
    load(['../../data/predict_search/' dataset '/predicted_specificity.mat']);
    y = y_pred(1, :); z = z_pred(1, :);
end

n_images = length(y);
test_size = 100:10:n_images;

for idx=1:length(test_size)

    fprintf('\nTest size = %d', test_size(idx));
    test_idx = randsample(n_images, test_size(idx));
    s_test = s(test_idx, test_idx);
    
    % BASELINE
    rank_b = zeros(1, test_size(idx));
    for query_idx = 1:length(test_idx)
        [~, idx_b] = sort(s_test(query_idx, :),'descend');
        rank_b(query_idx) = find(idx_b==query_idx);
    end
    eval(['rank_baseline.size_' num2str(test_size(idx)) '=rank_b;']);
    
    % GROUND TRUTH / PREDICTED SPECIFICITY
    r_s = zeros(test_size(idx), test_size(idx)); % scores
    y_out = y(test_idx); z_out = z(test_idx); % logit parameters
    rank_s = zeros(1, test_size(idx)); % ranks
    for query_idx=1:test_size(idx)
        progressbar(query_idx, 10, length(test_idx));
        
        for ref_idx=1:length(test_idx)
            r_s(query_idx, ref_idx) = glmval([y_out(ref_idx), z_out(ref_idx)]', s_test(query_idx, ref_idx), 'logit');
        end
        
        r_s(isnan(r_s(:))) = -Inf;
        
        [~, idx_s] = sort(r_s(query_idx, :), 'descend');
        rank_s(query_idx) = find(idx_s==query_idx);
        
    end
    eval(['rank_specificity.size_' num2str(test_size(idx)) '=rank_s;']);
    save(['../../data/predict_search/' dataset '/' method '_logistic.mat'], ...
         'rank_specificity', 'rank_baseline', 'test_size');
end

end