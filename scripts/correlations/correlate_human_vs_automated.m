% Interhuman correlation, correlation of automated vs. human specificity,
% histograms of specificity, and montage of different specificity values
%
% Author: Mainak Jas

clear all; close all;
addpath('../../library/export_fig/');

%% Load data %%
load('../../data/specificity_scores_MEM5S.mat');
load('../../data/specificity_automated.mat');
load('../../data/sentences/memorability_888_img_5_sent.mat');
load('../../data/target_images.mat');
load('../../data/memorability_mapping.mat');
S = load('../../data/specificity_alldatasets.mat');
P = load('../../data/specificity_scores_140315.mat');

%% Interhuman correlation %%
[ih_corr, ihpval] = corr(specificity(1:222), P.specificity, 'type', 'spearman');
fprintf('Correlation when 5 more sentences are added = %0.2f\n', ih_corr);

%% Correlate human specificity vs automated %%
[cr, pval] = corr(specificity, specificity_automated', 'type' ,'spearman');

scatter(specificity, specificity_automated', 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Measured specificity', 'Fontsize', 12); ylabel('Automated specificity', 'Fontsize', 12);

set(gca, 'XLim', [0 1], 'YLim', [0 1], 'Tickdir', 'out', ...
    'YTick',0:0.2:1, 'XTick', 0:0.2:1, 'TickLength', [0.005; 0.025]);
title(sprintf('Spearman''s \\rho = %0.2f, p-value < 0.01', cr), 'Fontsize', 14);

export_fig '../../plots/automated_vs_human.pdf' -transparent;

%% Show histogram of specifities %%
x_pos = [0.07, 0.55, 0.07, 0.55];
y_pos = [0.55, 0.55, 0.07, 0.07]; height = 0.38; width = 0.4;

figure; set(gcf, 'Position', [50, 300, 650, 600]); 
subplot(2,2,1, 'Position', [x_pos(1), y_pos(1), width, height]);

hist(specificity, 50);
set(gca, 'XTick', 0:0.2:1, 'TickLength', [0.005; 0.025], 'FontSize', 13);
title('MEM-5S (human)', 'Fontsize', 14);
ylabel('Number of images', 'Fontsize', 14);

datasets = {'memorability', 'pascal', 'clipart'};
titles = {'MEM-5S', 'MEM-5S (auto)', 'PASCAL-50S (auto)', 'ABSTRACT-50S (auto)'};
for i=2:4
    subplot(2,2,i, 'Position', [x_pos(i), y_pos(i), width, height]);
    eval(sprintf('hist(S.specificity.%s.mean, 25);', datasets{i-1}));

    set(gca, 'XTick', 0:0.2:1, 'TickLength', [0.005; 0.025], 'FontSize', 13, ...
        'XLim', [0 1]);
    if i>=3
        xlabel('Specificity value', 'Fontsize', 14);
    end

    if rem(i,2)
        ylabel('Number of images', 'Fontsize', 14);
        ylabh = get(gca, 'YLabel');
        set(ylabh,'Position',get(ylabh,'Position') + [0.03 0 0]);
    end
    
    title(titles{i});
end

export_fig '../../plots/specificity_histogram.pdf' -transparent;

%% Show montage of high/mid/low value specificity images %%
figure; [sorted_s, sorted_idx] = sort(specificity, 'descend');
pick_idx = [2, 202, 706, 887]; set(gcf, 'Position',  [364, 200, 1362, 500]);

for i=1:length(pick_idx)
    subtightplot(1,4,i);
    img_idx = sorted_idx(pick_idx(i));
    I = img(:, :, :, mapping(img_idx));
    imshow(I); title(sprintf('Specificity = %0.2f', specificity(img_idx)), 'Fontsize', 12);
    
    for j=1:size(memorability_sentences,2)
        text(0, 270 + j*15, memorability_sentences{img_idx, j}, 'Fontsize',14); 
    end
end
