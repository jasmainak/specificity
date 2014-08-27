% For pascal dataset

clear all;

% First generate pairs
comb = [];
for ref_idx=1:100
    for query_idx=ref_idx+1:50
        comb = [comb; [ref_idx, query_idx]];
    end
end

% Now select randomly 50 pairs
samples = comb(randsample(1225, 200), :);

samples = samples - 1; % convert to python compatible format

dlmwrite('choose_ref_query2.txt', samples, 'delimiter', '\t');