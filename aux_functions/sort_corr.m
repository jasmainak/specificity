function [r, idx] = sort_corr(feat_vec, metric)

r = corr(feat_vec, metric, 'type', 'spearman'); 

% NaNs occur if an object category doesn't come in any image
r(isnan(r)) = -999; 
[r, idx] = sort(r,'descend');

end

