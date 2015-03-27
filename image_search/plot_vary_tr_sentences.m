% Author : Mainak Jas
%
% Plots the retrival curve for Pascal and Clipart datasets

clear all; close all;

addpath(genpath('../../library/boundedline/'));
addpath('../../library/export_fig/');

X = load('../../data/search_results/pascal/ranks_baseline.mat', 'rank_b');
baseline_pascal = mean(X.rank_b);

X = load('../../data/search_results/clipart/ranks_baseline.mat', 'rank_b');
baseline_clipart = mean(X.rank_b);

ranks_pascal = [];
for run=1:25
    rank_ntr = [];
    for ntr=2:50
        X = load(sprintf('../../data/search_results/pascal/ranks_logistic_run%d_ntr%d.mat', run, ntr));
        %rank_s = cat(1, rank_s, X.rank_s);
        rank_ntr = cat(1, rank_ntr, X.rank_s);
    end
    
    %rank_pascal = mean(rank_s, 2);
    ranks_pascal = cat(3, ranks_pascal, rank_ntr);
end

rank_pascal = squeeze(mean(ranks_pascal, 2));

pascal_mean = mean(rank_pascal, 2);
pascal_std = std(rank_pascal, 0, 2);

rank_s = [];
for run = 1:25

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

boundedline(2:50, pascal_mean, pascal_std, 'b');
h1 = plot(2:50, pascal_mean, 'bo-', 'MarkerFaceColor', 'w');
plot(49, pascal_mean(end-1), 'go', 'MarkerFaceColor','g');
plot(50, pascal_mean(end), 'ko', 'MarkerFaceColor','k');
annotation('textarrow', [0.875, 0.875], [0.34, 0.26], 'Color', [0.5, 0.5, 0.5], 'Linewidth', 1);
text(49, 68, '2.5%', 'Fontsize', 14);
annotation('textarrow', [0.4, 0.27], [0.6, 0.35], 'Color', [0.5, 0.5, 0.5], 'Linewidth', 1);
text(13, 117, sprintf('specificity outperforms\n baseline after %d sentences', find(clipart_mean < baseline_clipart, 1, 'first') + 1), 'Fontsize', 14);

boundedline(2:48, clipart_mean, clipart_std, 'r');
h2 = plot(2:48, clipart_mean, 'rs-', 'MarkerFaceColor', 'w');
plot(47, clipart_mean(end-1), 'gs', 'MarkerFaceColor', 'g');
plot(48, clipart_mean(end), 'ks', 'MarkerFaceColor', 'k');
annotation('textarrow', [0.905, 0.905], [0.185, 0.125], 'Color', [0.5, 0.5, 0.5], 'Linewidth', 1);
text(50.5, 46, '1.0%', 'Fontsize', 14);
annotation('textarrow', [0.53, 0.4], [0.47, 0.2], 'Color', [0.5, 0.5, 0.5], 'Linewidth', 1);
text(21, 97.5, sprintf('specificity outperforms \nbaseline after %d sentences', find(pascal_mean < baseline_pascal, 1, 'first') + 1), 'Fontsize', 14);

h3 = plot([2, 50], [baseline_pascal, baseline_pascal],'b--');
h4 = plot([2, 48], [baseline_clipart, baseline_clipart], 'r--');

set(gca, 'TickDir','out','Box','off','XTick',[10:10:50], ...
    'XTickLabel',{'10C2','20C2','30C2','40C2','50C2'}, 'Fontsize',12, ...
    'TickLength', [0.007 0.007]);

legend([h1, h3, h2, h4], {'PASCAL-50S', 'Baseline', 'ABSTRACT-50S', ...
                          'Baseline'});
%title('Retrieval Curve'); 
ylabel('Mean rank of target image'); xlabel('number of training sentence pairs per image');

export_fig '../../plots/paper/vary_tr_sentences.pdf' -transparent;
