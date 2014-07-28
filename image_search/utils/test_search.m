clear all; close all;
load('../../../data/specificity_alldatasets.mat');

eval(['y = specificity.pascal.B0;']);
eval(['z = specificity.pascal.B1;']);

addpath('../../aux_functions/');  % for displaying progressbar
cd('../'); % for load_search_parameters
[~, ~, s, ~, ~, ~] = load_search_parameters('pascal');
cd('utils/');

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


rank_b = zeros(1, length(s));
for query_idx = 1:length(s)
    [~, idx_b] = sort(s(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end

save('test_search.mat', 'y', 'z', 's', 'rank_s', 'rank_b');
