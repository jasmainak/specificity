% CORRELATE_SCORES correlates specificity with sentence lengths,
% memorability, mean/median object area, object count, color & importance
%
% Input files
% -----------
%   specificity_scores_all.mat
%   memorability_mapping.mat
%   importance_scores.mat
%   memorability_sent_lengths.mat
%   target_features.mat (from Isola et al., CVPR 2011 memorability paper)
%   data/memorability/img_{%d}.jpg
%
% Output files
% ------------
%   correlate_specificity.pdf
%
% AUTHOR: Mainak Jas

%% Specify directories
close all; clear all;
addpath('../../library/export_fig/');

%% Load data
fprintf('Loading data ... ');
load('../../data/specificity_scores_MEM5S.mat');
load('../../data/memorability_mapping.mat');
load('../../data/importance_scores.mat');
load('../../data/memorability_sent_lengths.mat');
Feat = load('../../data/target_features.mat');
fprintf('[Done]\n');

%% Correlation with features

% SENTENCE LENGTHS
sent_lengths = double(cell2mat(sent_lengths));
[r.specVsMeanLength, pval1] = corr(specificity, mean(sent_lengths, 2), 'type', 'spearman');
fprintf('Correlation between specificity and mean sentence length = %0.2f\n', r.specVsMeanLength);

[r.specVsVaryLength, pval2] = corr(specificity, std(sent_lengths, 0, 2), 'type', 'spearman');
fprintf('Correlation between specificity and standard deviation in sentence length = %0.2f\n', r.specVsVaryLength);

% MEMORABILITY
[r.specVSmem, pval.specVsmem] = corr(mem(mapping),specificity,'type','spearman');
fprintf('Correlation of specificity with memorability = %0.2f\n', r.specVSmem);

areas = full(Feat.Areas);
areas(areas==0) = NaN; % compute mean and median by leaving out objects with 0 area

[max_area, max_idx] = nanmax(areas(:, mapping));

% OBJECT AREA
[r.medAreavsSpec, pval.medAreavsSpec] = corr(nanmedian(areas(:, mapping))', specificity, ...
                               'type','spearman');
fprintf('Correlation of specificity with median object area = %0.2f\n', r.medAreavsSpec);
r.meanAreavsSpec = corr(nanmean(areas(:, mapping))', specificity, ...
                       'type','spearman');
fprintf('Correlation of specificity with mean object area = %0.2f\n', r.meanAreavsSpec); 

% OBJECT COUNT
[r.objectcountVsSpec, pval.objectcountVsSpec] = corr(sum(Feat.Counts(:, mapping))', specificity, ...
                     'type','spearman');
fprintf('Correlation of specificity with object count = %0.2f\n', r.objectcountVsSpec);

% COLOR
for i=1:length(specificity)
    I = imread(sprintf('../../data/images/memorability/img_%d.jpg', i));
    
    red = I(:,:,1)./255; 
    green = I(:,:,2)./255; 
    blue = I(:,:,3)./255;
    
    rs(i) = mean(red(:)); gs(i) = mean(green(:)); bs(i) = mean(blue(:));
end

r.specVsred = corr(rs', specificity, 'type', 'spearman'); % correlation with red
fprintf('Correlation of specificity with red = %0.2f\n', r.specVsred);
r.specVsgreen = corr(gs', specificity, 'type', 'spearman'); % correlation with green
fprintf('Correlation of specificity with green = %0.2f\n', r.specVsgreen);
r.specVsblue = corr(bs', specificity, 'type', 'spearman'); % correlation with blue
fprintf('Correlation of specificity with blue = %0.2f\n', r.specVsblue);

% IMPORTANCE
min_area = 4000; % Same value as in Isola et al. NIPS memorability paper
object_pres = (full(Feat.Areas))>min_area;

% Truncate to only images for which specificity measurements are available
% only include objects for which there are at least 10 images
n_categories = size(object_pres,1);
include = zeros(1, n_categories);

for i=1:n_categories            
    if length(find(object_pres(i, mapping)==1))>=10
        include(i) = 1;
    end    
end
include = find(include);

object_pres = object_pres(include, mapping);
% Remove 'person sitting' category, because the importance score can be
% computed only for categories with one word in the name.
object_pres = object_pres([1:22, 24:41], :);
importance = importance([1:22, 24:41]);

% Correlation of importance with mean specificity of that object
for i=1:length(importance)    
    im_idx = find(object_pres(i, :)==1);
    obj_score(i) = mean(specificity(im_idx));
end

[r.importance, pval.importance] = corr(obj_score', importance, 'type', 'spearman');
fprintf('Correlation of specificity with object importance = %0.2f\n', r.importance);

%% Make figure

subplot(2,2,1);
scatter(mem(mapping), specificity, 5, 'filled'); h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Memorability score', 'Fontsize', 12); ylabel('Specificity score', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value < 0.01',r.specVSmem), 'Fontsize', 12);

subplot(2,2,2);
scatter(importance, obj_score', 5, 'filled');
xlabel('Importance score', 'Fontsize', 12); h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.importance, pval.importance), 'Fontsize', 12);

subplot(2,2,3);
scatter(nanmedian(areas(:, mapping))', specificity, 5, 'filled'); h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Median object area', 'Fontsize', 12); ylabel('Specificity score', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value < 0.01', r.medAreavsSpec), 'Fontsize', 12);

subplot(2,2,4);
scatter(sum(Feat.Counts(:, mapping))', specificity, 5, 'filled');
h = lsline; set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Object count', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value = %0.2f\n', r.objectcountVsSpec, pval.objectcountVsSpec), 'Fontsize', 12);

set(gcf, 'Position', [680   29   560   560]);
export_fig '../../plots/correlate_specificity.pdf' -transparent;