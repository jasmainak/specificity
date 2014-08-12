clear all;

addpath('utils/');
addpath('../aux_functions/');

load('search_parameters_pascal.mat');
load('../../data/predict_search/pascal/predicted_specificity_1000fold.mat');
load('../../data/predict_search/pascal/groundtruth_specificity.mat');

alphas = 0:0.1:1; rank_s = zeros(length(alphas), 1000);

poolobj = parpool;
for i=1:length(alphas)
    fprintf('alpha=%0.2f',alphas(i));
    %rank_s(i, :) = combine_specificity_search(s, y_pred, z_pred, 'weighted-score', alphas(i));
    rank_s(i, :) = combine_specificity_search(s, y_pred, z_pred, 'weighted-rank', alphas(i));
end
delete(poolobj);

% compare the three methods
rank_min = combine_specificity_search(s, y_pred, z_pred, 'min-rank', -999);
rank_b = baseline_search(s);
rank_spec = specificity_search(s, y_pred, z_pred);

delete(poolobj);

% plot results
plot(alphas, mean(rank_s,2), 'bo-', 'MarkerFaceColor', 'w');
xlabel('alpha', 'Fontsize', 12); ylabel('mean rank', 'Fontsize', 12);
title('Effect of alpha on mean rank (Pascal dataset / predicted specificity)', 'Fontsize', 14);
set(gca, 'box', 'off', 'tickdir', 'out');

[min_alpha, min_idx] = min(mean(rank_s, 2));
text(alphas(min_idx), min_alpha + 0.05, num2str(min_alpha));