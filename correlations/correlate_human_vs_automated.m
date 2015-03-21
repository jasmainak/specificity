clear all; close all;

addpath('../../library/export_fig/');

load('../../data/specificity_scores_all.mat');
S = load('../../data/specificity_scores_140315.mat');

%%%%% Interhuman correlation %%%%%

rng('default'); % to avoid surprises

% split 30 similarity scores into 2 parts 25 times
for i=1:25
    
    n_spec = size(scores,2)*size(scores,3);
    idx = randperm(n_spec);
    
    similarity_scores = reshape(scores, size(scores,1), n_spec);
    
    spec1 = mean(similarity_scores(:, idx(1:n_spec/2)), 2);
    spec2 = mean(similarity_scores(:, idx(n_spec/2 + 1:n_spec)), 2);
    
    sim_corr(i) = corr(spec1, spec2, 'type', 'spearman');
end

[ih_corr, ihpval] = corr(specificity(1:222), S.specificity, 'type', 'spearman');

%%%%% Correlate human specificity vs automated %%%%%

specificity_automated = zeros(size(specificity));

score_w = [];
for i=1:888
    load(sprintf('../../data/search_parameters/memorability/mus/img_img_%d.jpg.mat', i), 'scores_w');
    score_w = [score_w, scores_w];
    specificity_automated(i) = nanmean(scores_w);
end

[cr, pval] = corr(specificity, specificity_automated, 'type' ,'spearman');

scatter(specificity, specificity_automated, 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Measured specificity', 'Fontsize', 12); ylabel('Automated specificity', 'Fontsize', 12);

set(gca, 'XLim', [0 1], 'YLim', [0 1], 'Tickdir', 'out', ...
    'YTick',0:0.2:1, 'XTick', 0:0.2:1, 'TickLength', [0.005; 0.025]);
title(sprintf('Spearman''s \\rho = %0.2f, p-value < 0.01', cr), 'Fontsize', 14);

export_fig '../../plots/paper/automated_vs_human.pdf' -transparent;

%%%%% Show histogram of specifities %%%%%
x_pos = [0.07, 0.55, 0.07, 0.55];
y_pos = [0.55, 0.55, 0.07, 0.07]; height = 0.38; width = 0.4;

figure; set(gcf, 'Position', [50, 300, 650, 600]); 
subplot(2,2,1, 'Position', [x_pos(1), y_pos(1), width, height]);

hist(specificity, 50);
h = findobj(gca,'Type','patch');
%set(h,'Facecolor','b', 'Linestyle', 'none');

set(gca, 'XTick', 0:0.2:1, 'TickLength', [0.005; 0.025], 'FontSize', 13);
title('MEM-5S (human)', 'Fontsize', 14);
ylabel('Number of images', 'Fontsize', 14);

S = load('../../data/specificity_alldatasets.mat');
datasets = {'memorability', 'pascal', 'clipart'};
titles = {'MEM-5S', 'MEM-5S (auto)', 'PASCAL-50S (auto)', 'ABSTRACT-50S (auto)'};
for i=2:4
    subplot(2,2,i, 'Position', [x_pos(i), y_pos(i), width, height]);
    eval(sprintf('hist(S.specificity.%s.mean, 25);', datasets{i-1}));
    h = findobj(gca,'Type','patch');
    %set(h,'Facecolor','b', 'Linestyle', 'none');

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

export_fig '../../plots/paper/specificity_histogram.pdf' -transparent;

%%%%% Show montage of high/mid/low value specificity images %%%%

figure;
addpath('../aux_functions/');
load('../../data/sentences/memorability_888_img_5_sent.mat');
[sorted_s, sorted_idx] = sort(specificity, 'descend');

pick_idx = [2, 202, 706, 887]; set(gcf, 'Position',  [364, 200, 1362, 500]);

for i=1:length(pick_idx)
    subtightplot(1,4,i);
    img_idx = sorted_idx(pick_idx(i));
    I = imread(sprintf('../../data/images/memorability/img_%d.jpg', img_idx));
    imshow(I); title(sprintf('Specificity = %0.2f', specificity(img_idx)), 'Fontsize', 12);
    
    for j=1:size(memorability_sentences,2)
        text(0, 270 + j*15, memorability_sentences{img_idx, j}, 'Fontsize',14); 
    end
end
