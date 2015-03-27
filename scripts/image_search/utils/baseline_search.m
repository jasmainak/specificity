function rank_b = baseline_search(s)
% s : 2D array
%   similarity score between query and reference sentence

    rank_b = zeros(1, length(s));
    for query_idx = 1:length(s)
        [~, idx_b] = sort(s(query_idx, :),'descend');
        rank_b(query_idx) = find(idx_b==query_idx);
    end
end