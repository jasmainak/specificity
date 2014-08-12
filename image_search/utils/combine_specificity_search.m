function rank_s = combine_specificity_search(s, y, z, method, alpha)
% s : 2D array (n_images x n_images)
%   similarity score between query and reference sentence
% y : 1D array (n_images x 1)
%   first parameter of logistic regression
% z : 1D array (n_images x 1)
%   second parameter of logistic regression
% method : str
%   'weighted-score' | 'max-score' | 'weighted rank' | 'min-rank'
% alpha : float
%   0 <= alpha <=1 for weighted-* methods

    rank_s = zeros(1, length(s)); % ranks
    p_s = zeros(length(s), length(s)); % logit scores
    r_s = zeros(length(s), length(s)); % combined scores
    for query_idx=1:length(s)
        progressbar(query_idx, 10, length(s));

        % Calculate logit score
        parfor ref_idx=1:length(s)
            p_s(query_idx, ref_idx) = glmval([y(ref_idx), z(ref_idx)]', s(query_idx, ref_idx), 'logit');
        end

        % Combine scores
        if strcmpi(method, 'weighted-score')
            r_s(query_idx, :) = alpha*s(query_idx, :) + (1-alpha)*p_s(query_idx, :);
        elseif strcmpi(method, 'max-score')
            r_s(query_idx, :) = nanmax(s(query_idx, :), p_s(query_idx, :));
        end

        if regexpi(method, 'rank')
            [~, idx_p] = sort(p_s(query_idx, :), 'descend');
            [~, idx_b] = sort(s(query_idx, :), 'descend');

            [~, rank_p] = sort(idx_p, 'ascend');
            [~, rank_b] = sort(idx_b, 'ascend');
        elseif regexpi(method, 'score')
            r_s(query_idx, :) = p_s(query_idx, :);
            r_s(isnan(r_s(:))) = -Inf;
            [~, idx_s] = sort(r_s(query_idx, :), 'descend');
        end

        % combine ranks
        if strcmpi(method, 'weighted-rank')
            rank = alpha*rank_b + (1-alpha)*rank_p;
            [~, idx_s] = sort(rank, 'ascend');
        elseif strcmpi(method, 'min-rank')
            rank = min(rank_b, rank_p);
            [~, idx_s] = sort(rank, 'ascend');
        end

        % find rank of query image
        rank_s(query_idx) = find(idx_s==query_idx);

    end
end
