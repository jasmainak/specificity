clear all;
load('../../data/specificity_scores_all.mat');

%% First check if there are any clashes (same worker on both sides of the split)
% sanity check
%workers(1, 2, 3) = workers(1, 2, 1);
%workers(2, 1, 2) = workers(2, 1, 1);
%workers(1, 5, 1) = workers(1, 5, 3);

split1 = 2; split2 = setdiff(1:3, split1);

% find frequencies of each worker / image
freqs = zeros(size(scores));
for img_idx = 1:size(scores,1)
    for pair_idx = 1:size(scores,2)
        for assignment_idx = 1:size(scores,3)
            freqs(img_idx, pair_idx, assignment_idx) = length(strmatch(workers(img_idx, pair_idx, assignment_idx), workers(img_idx, :), 'exact'));
        end
    end
end

new_workers = cell(size(scores));
new_scores = zeros(size(scores));

for img_idx = 1:size(scores, 1)
    
    for pair_idx = 1:size(scores, 2)
        [~, max_idx] = min(freqs(img_idx, pair_idx, :));
        new_workers(img_idx, pair_idx, 1) = workers(img_idx, pair_idx, max_idx);
        new_workers(img_idx, pair_idx, 2:3) = workers(img_idx, pair_idx, setdiff(1:3, max_idx));
        new_scores(img_idx, pair_idx, 1) = scores(img_idx, pair_idx, max_idx);
        new_scores(img_idx, pair_idx, 2:3) = scores(img_idx, pair_idx, setdiff(1:3, max_idx));
    end
end

% Count the number of clashes between the two splits

clashes = zeros(size(scores,1), 1);

for img_idx = 1:size(scores, 1)
    worker_split1 = squeeze(new_workers(img_idx, :, split1))';
    worker_split2 = squeeze(new_workers(img_idx, :, split2));
    clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
end

mean(clashes)

specificity_split1 = mean(mean(new_scores(:, :, split1), 3), 2);
specificity_split2 = mean(mean(new_scores(:, :, split2),3),2);

corr(specificity_split1, specificity_split2, 'type', 'spearman')

% cant_swap = 0;
% for img_idx = 1:size(scores,1)
%     
%     worker_split1 = squeeze(workers(img_idx, :, 3))';
%     worker_split2 = squeeze(workers(img_idx, :, [1,2]));
%     
%     clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
%     
%     while clashes(img_idx) > 0
%         
%         fprintf('Image %d', img_idx);
%         fprintf(' ... %d clashes\n', clashes(img_idx));
%         
%         for pair_idx = 1:size(scores,2)
%             if strmatch(worker_split1{pair_idx}, worker_split2, 'exact')
%                 
%                 fprintf('\tClash with [%d] %s\n', pair_idx, worker_split1{pair_idx});
%                 pause(1.0);
%                 
%                 % swap
%                 swap_idx = randi(2);               
%                 if length(strmatch(worker_split2(pair_idx, swap_idx), worker_split2)) > 1
%                     fprintf('\tCan''t swap\n');
%                     cant_swap = 1; continue;
%                 end
%                 
%                 [worker_split1(pair_idx), worker_split2(pair_idx, swap_idx)] = swap(worker_split1(pair_idx), worker_split2(pair_idx, swap_idx));
%             elseif cant_swap
%                 fprintf('\tGetting here because couldn''t swap ... \n'); 
%                 if rand > 0.5
%                     [worker_split1(pair_idx), worker_split2(pair_idx, swap_idx)] = swap(worker_split1(pair_idx), worker_split2(pair_idx, swap_idx));
%                 end
%                 cant_swap = 0;
%             end
%             clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
%         end
%     end
% end

% 
% function [v1, v2] = swap(v1, v2)
% 
%     temp = v1;
%     v1 = v2;
%     v2 = temp;
% 
% end