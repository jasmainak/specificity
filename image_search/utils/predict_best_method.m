function [rank_best, rank_min, rank_s, rank_b] = predict_best_method(s, y, z, feat)

rng('default'); % To avoid surprises in the future
addpath('../aux_functions/');
addpath(genpath('../../library/libsvm-3.17/'));

[rank_min, rank_s, rank_b] = rank_all_score(s, y, z);

% get groundtruth
ranks = [rank_b; rank_min; rank_s];
rank_best = zeros(1, length(ranks));

% get features and labels
[~, b_method] = min(ranks);

% grid search
% just use default parameters for now
% bestc = grid_search_linear(feat, b_method) * 800/1000;

idx = crossvalind('Kfold', length(b_method), 5);

for fold_idx=1:5
    train_idx = find(idx ~= fold_idx); test_idx = find(idx == fold_idx);

    [Z_train,mu,sigma] = zscore(feat(train_idx,:));
    %model = svmtrain2(b_method(train_idx)', Z_train, ['-s 0 -q -t 0 -c ' num2str(bestc)]);
    model = svmtrain2(b_method(train_idx)', Z_train, '-s 0 -q -t 0');

    sigma0 = sigma;
    sigma0(sigma0==0) = 1;
    Z_test = bsxfun(@minus,feat(test_idx,:), mu);
    Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
    y_out = svmpredict2(b_method(test_idx)', Z_test, model, '-q');
    rank_best(test_idx) = ranks(sub2ind(size(ranks), y_out, test_idx));

end

end

function bestc = grid_search_linear(X, y)

bestcv = 0; bestc = 2^-1;
for log2c = -1:3
    cmd = ['-s 0 -t 0 -q -v 5 -c ', num2str(2^log2c)];
    cv = svmtrain2(X, y, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
end

end

function [rank_min, rank_s, rank_b] = rank_all_score(s, y, z)
    rank_s = zeros(1, length(s)); % ranks
    rank_min = zeros(1, length(s));
    r_s = zeros(length(s), length(s)); % logit scores
    for query_idx=1:length(s)
        progressbar(query_idx, 10, length(s));

        % Calculate logit score
        parfor ref_idx=1:length(s)
            r_s(query_idx, ref_idx) = glmval([y(ref_idx), z(ref_idx)]', s(query_idx, ref_idx), 'logit');
        end

        [~, idx_s] = sort(r_s(query_idx, :), 'descend');
        [~, idx_b] = sort(s(query_idx, :), 'descend');

        [~, spec_ranks] = sort(idx_s, 'ascend');
        [~, base_ranks] = sort(idx_b, 'ascend');

        rank = min(base_ranks, spec_ranks);
        [~, idx_min] = sort(rank, 'ascend');

        % find rank of query image
        rank_min(query_idx) = find(idx_min==query_idx);
        rank_s(query_idx) = find(idx_s==query_idx);
        rank_b(query_idx) = find(idx_b==query_idx);
    end
end

