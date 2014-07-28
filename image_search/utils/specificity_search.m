function rank_s = specificity_search(s, y, z)
% s : 2D array (n_images x n_images)
%   similarity score between query and reference sentence
% y : 1D array (n_images x 1)
%   first parameter of logistic regression
% z : 1D array (n_images x 1)
%   second parameter of logistic regression

    addpath('../../aux_functions/');  % for displaying progressbar

    rank_s = zeros(1, length(s)); % ranks
    r_s = zeros(length(s), length(s)); % scores
    for query_idx=1:length(s)
        progressbar(query_idx, 10, length(s));

        for ref_idx=1:length(s)
            r_s(query_idx, ref_idx) = glmval([y(ref_idx), z(ref_idx)]', s(query_idx, ref_idx), 'logit');
        end

        r_s(isnan(r_s(:))) = -Inf;

        [~, idx_s] = sort(r_s(query_idx, :), 'descend');
        rank_s(query_idx) = find(idx_s==query_idx);

    end
end
