
function correlate_interhuman()
load('../../data/specificity_scores_all.mat');

%% First check if there are any clashes (same worker on both sides of the split)
% sanity check
%workers(1, 2, 3) = workers(1, 2, 1);
%workers(2, 1, 2) = workers(2, 1, 1);
%workers(1, 5, 1) = workers(1, 5, 3);

clashes = zeros(size(scores,1), 1);

for img_idx = 1:size(scores,1)
    
    worker_split1 = squeeze(workers(img_idx, :, 3))';
    worker_split2 = squeeze(workers(img_idx, :, [1,2]));
    
    clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
    
    tic;
    while clashes(img_idx) > 0
        
        fprintf('Image %d', img_idx);
        fprintf(' ... %d clashes\n', clashes(img_idx));
        
        for pair_idx = 1:size(scores,2)
            if strmatch(worker_split1{pair_idx}, worker_split2, 'exact')
                
                fprintf('\t[%0.2f] Clash with [%d] %s\n', toc, pair_idx, worker_split1{pair_idx});
                pause(1.0);
                
                % swap
                swap_idx = randi(2);               
                %if length(strmatch(worker_split2(pair_idx, swap_idx), worker_split2)) > 1
                    
                %end
                [worker_split1(pair_idx), worker_split2(pair_idx, swap_idx)] = swap(worker_split1(pair_idx), worker_split2(pair_idx, swap_idx));
                
            end
            clashes(img_idx) = length(find(ismember(worker_split2(:), worker_split1)));
        end
    
    if toc > 50
        fprintf('40 seconds elapsed \n');
        break;
    end
        
    end
end

end

function [v1, v2] = swap(v1, v2)

    temp = v1;
    v1 = v2;
    v2 = temp;

end