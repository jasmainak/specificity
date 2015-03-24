% Script to split the similarity ratings so that there are minimum
% number of clashes in the splits & then find the inter-human correlation.
% Author: Mainak Jas

clear all;
load('../../data/specificity_scores_all.mat');

% find frequencies of each worker / image
new_scores = zeros(size(scores));
new_workers = cell(size(workers));
for img_idx = 1:size(scores,1)
    
    old_workers = squeeze(workers(img_idx, :, :));
    new_worker = cell(size(scores,2), size(scores,3));
    new_score = -1*ones(size(new_worker));
    
    % find unique workers
    unique_workers.name = unique(workers(img_idx, :));
    unique_workers.freq = zeros(1, length(unique_workers.name));
    for worker_idx = 1:length(unique_workers.name)
        unique_workers.freq(worker_idx) = length(strmatch(unique_workers.name(worker_idx), workers(img_idx, :), 'exact'));
    end

    % sort workers by frequency
    [unique_workers.freq, sort_idx] = sort(unique_workers.freq, 'descend');
    unique_workers.name = unique_workers.name(sort_idx);

    for worker_idx = 1:length(unique_workers.name)
        
        % Find where in the old matrix the workers reside
        ind = strmatch(unique_workers.name(worker_idx), workers(img_idx, :), 'exact');
        [pair_idx, assignment_idx] = ind2sub(size(new_worker), ind);
        
        % Find assignment with the most empty spots that the worker can
        % fill
        empty_spots = zeros(3, 1);
        for i=1:3
            empty_spots(i) = length(find(new_score(pair_idx, i) == -1));
        end
        [~, column_idx] = max(empty_spots);
        
        for i=1:length(pair_idx)
            
            % Put the score in if it's empty
            if new_score(pair_idx(i), column_idx) == -1
                new_worker(pair_idx(i), column_idx) = old_workers(pair_idx(i), assignment_idx(i));
                new_score(pair_idx(i), column_idx) = scores(img_idx, pair_idx(i), assignment_idx(i));
            % Otherwise find some other assignment
            else
                column_idxs = find(new_score(pair_idx(i), :) == -1);
                new_worker(pair_idx(i), column_idxs(1)) = old_workers(pair_idx(i), assignment_idx(i));
                new_score(pair_idx(i), column_idxs(1)) = scores(img_idx, pair_idx(i), assignment_idx(i));
            end
        end
    end
    
    new_scores(img_idx, :, :) = new_score;
    new_workers(img_idx, :, :) = new_worker;
    
end

% Find mean number of clashes

corrs_888 = zeros(3,1);
corrs_222 = zeros(3,1);

for split1 = 1:3
    split2 = setdiff(1:3, split1);
    clashes = zeros(size(scores,1), 1);

    for img_idx = 1:size(scores, 1)
        worker_split1 = squeeze(new_workers(img_idx, :, split1))';
        worker_split2 = squeeze(new_workers(img_idx, :, split2));
        clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
    end

    mean(clashes);

    % Find split specificity scores

    specificity_split1 = mean(mean(new_scores(:, :, split1), 3), 2);
    specificity_split2 = mean(mean(new_scores(:, :, split2),3),2);

    corrs_888(split1) = corr(specificity_split1, specificity_split2, 'type', 'spearman');
    corrs_222(split1) = corr(specificity_split1(1:222), specificity_split2(1:222), 'type', 'spearman');
end

fprintf('Consistency across subjects (888 images) = %0.2f\n', mean(corrs_888));
fprintf('Consistency across subjects (222 images) = %0.2f\n', mean(corrs_222));