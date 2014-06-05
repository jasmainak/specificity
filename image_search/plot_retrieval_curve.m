% Author : Mainak Jas
%
% Plots the retrival curve for Pascal and Clipart datasets

clear all; close all;

addpath(genpath('../../library/boundedline/'));

X = load('../../data/search_results/pascal/ranks_baseline.mat', 'rank_b');
baseline_pascal = mean(X.rank_b);

X = load('../../data/search_results/clipart/ranks_baseline.mat', 'rank_b');
baseline_clipart = mean(X.rank_b);

rank_s = [];
for ntr=2:50
    X = load(sprintf('../../data/search_results/pascal/ranks_logistic_run1_ntr%d.mat', ntr));
    rank_s = cat(1, rank_s, X.rank_s);
end

rank_pascal = mean(rank_s, 2);

rank_s = [];
for run = 1:3

    rank_ntr = [];
    for ntr=2:48
        X = load(sprintf('../../data/search_results/clipart/ranks_logistic_run%d_ntr%d.mat', run, ntr));
        rank_ntr = cat(1, rank_ntr, X.rank_s);
    end

    rank_s = cat(3, rank_s, rank_ntr);

end

rank_clipart = squeeze(mean(rank_s, 2));

clipart_mean = mean(rank_clipart, 2);
clipart_std =  std(rank_clipart, 0, 2);

% Do the actual plotting

h1 = plot(2:50, rank_pascal, 'b'); hold on;
plot(2:50, rank_pascal, 'bo', 'MarkerFaceColor', 'w');
plot(49, rank_pascal(end-1), 'go', 'MarkerFaceColor','g');
plot(50, rank_pascal(end), 'ko', 'MarkerFaceColor','k');

boundedline(2:48, clipart_mean, clipart_std, 'r');
h2 = plot(2:48, clipart_mean, 'r');
plot(2:48, clipart_mean, 'ro', 'MarkerFaceColor', 'w');
plot(47, clipart_mean(end-1), 'go', 'MarkerFaceColor', 'g');
plot(48, clipart_mean(end), 'ko', 'MarkerFaceColor', 'k');

h3 = plot([2, 50], [baseline_pascal, baseline_pascal],'k--');
plot([2, 48], [baseline_clipart, baseline_clipart], 'k--');

set(gca, 'TickDir','out','Box','off','XTick',[10:10:50], ...
    'XTickLabel',{'10C2','20C2','30C2','40C2','50C2'}, 'Fontsize',12);

legend([h1, h2, h3], {'Pascal (50 sentences)', 'Clipart (48 sentences)', ...
                      'Baselines'});
title('Retrieval Curve'); ylabel('Mean rank of target image'); xlabel('# training sentences');
