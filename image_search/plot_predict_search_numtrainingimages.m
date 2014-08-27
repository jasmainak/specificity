clear all;
dataset = 'pascal'; runs = 20;
features = 'decaf';
train_size = [10:10:800];

addpath(genpath('../../library/boundedline/'));
load('../../data/predict_search/pascal/search_baseline.mat', 'rank_b');

rank_s = [];
for run_idx=1:runs

    predictor = 'logistic';
    filename = sprintf('../../data/predict_search/%s/search_run%d_%s_%s.mat', ...
        dataset, run_idx, features, predictor);
    mat = load(filename);

    rank_s = cat(3, rank_s, mat.rank_s);

end

predictor = 'groundtruth'; run_idx = 1;
%filename = '../../data/predict_search/pascal/search_run1_decaf_groundtruth.mat';
filename = sprintf('../../data/predict_search/%s/search_run%d_%s_%s.mat', ...
    dataset, run_idx, features, predictor);
mat = load(filename);

rank_s_groundtruth = mean(mean(mat.rank_s));

meanrank_s = squeeze(mean(rank_s, 2));
meanground_s = squeeze(mean(rank_s_groundtruth, 2));

boundedline(train_size, mean(meanrank_s, 2), std(meanrank_s, 0, 2)); hold on;
h1 = plot(train_size, mean(meanrank_s, 2), 'bo-', 'MarkerFaceColor', 'w', 'Markersize', 6);
h2 = plot([10, 800], [mean(rank_b,2), mean(rank_b,2)], 'r--');
legend([h1, h2], {'predicted specificity', 'baseline'}, ...
       'location', 'northeast');
title(sprintf('Increasing training size (groundtruth=%0.2f, runs=%d)',rank_s_groundtruth, runs), 'Fontsize', 14);
ylabel('Mean rank', 'Fontsize', 12); xlabel('Training size (200 images ranked)', 'Fontsize', 12);
set(gca,'tickdir', 'out', 'box', 'off');
