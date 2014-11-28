clear all;

load('../../data/specificity_alldatasets.mat');
load('../../data/image_features/feat_clipart.mat', 'Feat', 'Feat_names');

thresh = 0.05;

y = specificity.clipart.mean;
y_nanfree = y(~isnan(y));

Feat = rmfield(Feat, {'x', 'y', 'z', 'flip'});
fdnames = fieldnames(Feat);

for idx = 1:length(fdnames)
    fprintf(['\n' upper(fdnames{idx}) '\n']);
    fprintf('==================\n');
    
    feat = Feat.(fdnames{idx});
    feat_names = Feat_names.(fdnames{idx});

    feat(feat(:)<0.05) = 0;  % Clamp small values to zero
    
    cr = zeros(1, size(feat,2));
    for i=1:size(feat, 2)
        cr(i) = corr(y_nanfree, feat(~isnan(y), i), 'type', 'spearman');
    end
    
    % Remove NaN correlations
    feat_names = feat_names(~isnan(cr));
    cr = cr(~isnan(cr));
    
    [cr_sorted, sorted_idx] = sort(cr, 'descend');
    for i=1:min(length(cr_sorted), 10)
        if cr_sorted(i) > 0.1
            fprintf('%0.3f ... %s\n', cr_sorted(i), feat_names{sorted_idx(i)});
        end
    end
end