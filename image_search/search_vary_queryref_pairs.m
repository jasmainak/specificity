clear all;

s_dir = '../../data/search_parameters/pascal/s/';
addpath('utils/'); addpath('../aux_functions');

load('../../data/predict_search/pascal/predicted_specificity_1000fold.mat');
load([s_dir 'combined_similarity.mat'], 'similarity');

for i=1:length(similarity)
    fprintf('\nPair %d ', i);
    rank_b(i, :) = baseline_search(similarity(i).s);
    fprintf('baseline = %0.2f / %0.2f', mean(rank_b(i, :)), mean(rank_b(:)));
    rank_s(i, :) = specificity_search(similarity(i).s, y_pred, z_pred);
    fprintf('specificity = %0.2f / %0.2f', mean(rank_s(i, :)), mean(rank_s(:)));
    fprintf('[Done]');
end

% Plot the results
for i=1:length(similarity)
    cum_rank = rank_b(1:i, :);
    cummean_b(i) = mean(cum_rank(:));
    cum_rank = rank_s(1:i, :);
    cummean_s(i) = mean(cum_rank(:));
end

subplot(1,2,1);
plot(cummean_b, 'bo-', 'MarkerFaceColor', 'w', 'MarkerSize', 6); hold on; 
plot(cummean_s, 'ro-', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
ylabel('Cumulative mean rank', 'Fontsize', 12);
xlabel('Number of query-ref pairs', 'Fontsize', 12);
set(gca, 'Tickdir', 'out', 'box', 'off');
legend('baseline', 'specificity', 'location', 'southeast');

subplot(1,2,2);
plot(cummean_b - cummean_s, 'bo-', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
ylabel('Cumulative mean rank difference(baseline - specificity)', 'Fontsize', 12);
xlabel('Number of query-ref pairs', 'Fontsize', 12);
set(gca, 'Tickdir', 'out', 'box', 'off');
