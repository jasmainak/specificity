% Find various correlations

%% Specify directories
close all; clear all;

addpath('../aux_functions');
addpath('../../library/export_fig/');

proj_dir = '../../';
img_dir = [proj_dir 'library/cvpr_memorability_data/Data/Image data'];

%% Load data
fprintf('Loading data ... ');
load('../../data/specificity_scores_all.mat');
load('../../data/memorability_mapping.mat');
Feat = load([img_dir '/target_features.mat']);
fprintf('[Done]\n');

fprintf('Loading images ... ');

if ~exist('img','var')    
%    load([img_dir '/target_images.mat']);
    size_img = 256;
end

fprintf('[Done]\n');

%% Correlation between specificity and memorability
[r.specVSmem, pval.specVsmem] = corr(mem(mapping),specificity,'type','spearman');

subplot(2,2,1);
scatter(mem(mapping), specificity, 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Memorability score', 'Fontsize', 12); ylabel('Specificity score', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value < 0.01',r.specVSmem), 'Fontsize', 12);

%% Consistency Analysis

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

r.interhuman_sim_mean = mean(sim_corr);
r.interhuman_sim_std = std(sim_corr);

%% Correlation with features

% Correlation of memorability and specificity with object areas, 
% object counts & object presence

min_area = 4000; % Same value as in Isola et al. NIPS memorability paper
object_pres = (full(Feat.Areas))>min_area;

[r.areasVsMem, idx.am] = sort_corr(Feat.Areas(:, mapping)', mem(mapping));
[r.areasVsSpec, idx.as] = sort_corr(Feat.Areas(:, mapping)', specificity);
[r.countsVsMem, idx.cm] = sort_corr(Feat.Counts(:, mapping)', mem(mapping));
[r.countsVsSpec, idx.cs] = sort_corr(Feat.Counts(:, mapping)', specificity);
[r.presVsMem, idx.pm] = sort_corr(object_pres(:, mapping)', mem(mapping));
[r.presVsSpec, idx.ps] = sort_corr(object_pres(:, mapping)', specificity);

% Display top-10 correlations (category specific)

fprintf('\nObject areas with Memorability\n\n');
disp_corr(r.areasVsMem, idx.am, Feat.objectnames, 10);
fprintf('\nObject areas with Specificity\n\n');
disp_corr(r.areasVsSpec, idx.as, Feat.objectnames, 10);
fprintf('\nObject counts with Memorability\n\n');
disp_corr(r.countsVsMem, idx.cm, Feat.objectnames, 10);
fprintf('\nObject counts with Specificity\n\n');
disp_corr(r.countsVsSpec, idx.cs, Feat.objectnames, 10);
fprintf('\nObject presence with Memorability\n\n');
disp_corr(r.presVsMem, idx.pm, Feat.objectnames, 10);
fprintf('\nObject presence with Specificity\n\n');
disp_corr(r.presVsSpec, idx.ps, Feat.objectnames, 10);

% Generic correlations

y = full(Feat.Areas);
y(y==0) = NaN; % compute mean and median by leaving out objects with 0 area

[max_area, max_idx] = nanmax(y(:, mapping));

r.maxAreaVsSpec = corr(max_area', specificity, ...
                       'type','spearman');
[r.medAreavsSpec, pval.medAreavsSpec] = corr(nanmedian(y(:, mapping))', specificity, ...
                               'type','spearman');
r.medAreaVsCount = corr(nanmedian(y(:, mapping))', sum(Feat.Counts(:, mapping))');

subplot(2,2,3);
scatter(nanmedian(y(:, mapping))', specificity, 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Median object area', 'Fontsize', 12); ylabel('Specificity score', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value < 0.01', r.medAreavsSpec), 'Fontsize', 12);

                   
r.meanAreavsSpec = corr(nanmean(y(:, mapping))', specificity, ...
                       'type','spearman');

%for i=1:length(max_idx)
%    y(max_idx(i), mapping) = NaN; % Check carefully
%end
%second_max = nanmax(y

clear y;
                   
[r.objectcountVsSpec, pval.objectcountVsSpec] = corr(sum(Feat.Counts(:, mapping))', specificity, ...
                     'type','spearman');
subplot(2,2,4);
scatter(sum(Feat.Counts(:, mapping))', specificity, 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('Object count', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.objectcountVsSpec, pval.objectcountVsSpec), 'Fontsize', 12);

% Correlate object distribution with specificity

for i=1:length(mapping) % Iterate over images
    objarray = Feat.Dmemory(mapping(i)).annotation.object;
    u = 1;
    for j=1:length(objarray) % Iterate over objects
        x = objarray(j).polygon.x;
        y = objarray(j).polygon.y;
        
        if polyarea(x,y)>min_area
            geom = polygeom(x,y);
            x_cen(u) = geom(2); y_cen(u) = size_img - geom(3) + 1; % matlab y-coordinates go in opposite direction
            u = u+1;
        end
        
        scatter_x(i) = std(x_cen); scatter_y(i) = std(y_cen);
        mean_x(i) = mean(x_cen); mean_y(i) = mean(y_cen);
    end
end

[r.scatterxVsSpec, pval.scatterxVsSpec] = corr(scatter_x',specificity,'type','spearman');
subplot(4,2,5);
scatter(scatter_x', specificity, 5, 'filled');
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
xlabel('object scatter in x-direction', 'Fontsize', 12);
ylabel('Specificity score', 'Fontsize', 12);
title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.scatterxVsSpec, pval.scatterxVsSpec), 'Fontsize', 12);

[r.scatteryVsSpec, pval.scatteryVsSpec] = corr(scatter_y',specificity,'type','spearman');
% subplot(4,2,6);
% scatter(scatter_y', specificity, 5, 'filled');
% h = lsline;
% set(h, 'Color', 'r', 'linewidth', 2);
% xlabel('object scatter in y-direction', 'Fontsize', 12);
% title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.scatteryVsSpec, pval.scatteryVsSpec), 'Fontsize', 12);

[r.meanxVsSpec, pval.meanxVsSpec] = corr(mean_x',specificity,'type','spearman');
% subplot(4,2,7);
% scatter(mean_x', specificity, 5, 'filled');
% h = lsline;
% set(h, 'Color', 'r', 'linewidth', 2);
% ylabel('Specificity score', 'Fontsize', 12);
% xlabel('mean x-coordinate of objects', 'Fontsize', 12);
% title(sprintf('\\rho = %0.2f, p-value < 0.01', r.meanxVsSpec), 'Fontsize', 12);

[r.meanyVsSpec, pval.meanyVsSpec] = corr(mean_y',specificity,'type','spearman');
% subplot(4,2,8);
% scatter(mean_y', specificity, 5, 'filled');
% xlabel('mean y-coordinate of objects', 'Fontsize', 12);
% h = lsline;
% set(h, 'Color', 'r', 'linewidth', 2);
% title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.meanyVsSpec, pval.meanyVsSpec), 'Fontsize', 12);

%% IMPORTANCE

load('../../data/importance_scores.mat');
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
% Remove 'person sitting' category
object_pres = object_pres([1:22, 24:41], :);
importance = importance([1:22, 24:41]);

% Correlation of importance with mean specificity of that object
for i=1:length(importance)    
    im_idx = find(object_pres(i, :)==1);
    obj_score(i) = mean(specificity(im_idx));
end

[r.importance, pval.importance] = corr(obj_score', importance, 'type', 'spearman');
subplot(2,2,2);
scatter(importance, obj_score', 5, 'filled');
xlabel('Importance score', 'Fontsize', 12);
h = lsline;
set(h, 'Color', 'r', 'linewidth', 2);
title(sprintf('\\rho = %0.2f, p-value = %0.2f', r.importance, pval.importance), 'Fontsize', 12);

set(gcf, 'Position', [680   29   560   560]);

export_fig '../../plots/paper/correlate_specificity.pdf' -transparent;

break;
%% Show images of a particular category

categ = 'platform';

categ_idx = find(strcmp(Feat.objectnames, categ)>0);
image_idx = find(object_pres(categ_idx, :)>0);
[~, used_idx] = intersect(image_idx, mapping);

for i=1:length(image_idx)
    subtightplot(3,2,i);
    imshow(img(:,:,:,image_idx(i)));
    
    if find(used_idx==i)>0
       w = 3;
       hold on; plot([w 256 256 w w], [w w 256 256 w],'r', 'Linewidth',w); 
    end
end