% Find various correlations

%% Specify directories
close all; 
clearvars -except img Feat specificity scores mem mapping;

addpath('aux_functions');

proj_dir = '/home/mainak/Desktop/specificity';
img_dir = [proj_dir '/library/cvpr_memorability_data/Data/Image data'];

%% Load data
fprintf('Loading data ... ');

if ~exist('specificity','var')
    load('../data/specificity_scores.mat');
end

if ~exist('mem','var')
    load('../data/memorability_mapping.mat');
end

if ~exist('Feat','var')
    Feat = load([img_dir '/target_features.mat']);
end

fprintf('[Done]\n');

fprintf('Loading images ... ');

if ~exist('img','var')    
    load([img_dir '/target_images.mat']);
end

fprintf('[Done]\n');

%% Correlation between specificity and memorability
r.specVSmem = corr(mem(mapping),specificity,'type','spearman');

%% Consistency Analysis

% Compute partial specificity scores
splits.spec12 = mean(mean(scores(:,:,1:2),3),2);
splits.spec23 = mean(mean(scores(:,:,2:3),3),2);
splits.spec13 = mean(mean(scores(:,:,[1,3]),3),2);

splits.spec1 = mean(mean(scores(:,:,1),3),2);
splits.spec2 = mean(mean(scores(:,:,2),3),2);
splits.spec3 = mean(mean(scores(:,:,3),3),2);

% Correlation between partial specificity scores
r.spec23VSspec1 = corr(splits.spec23, splits.spec1,'type','spearman');
r.spec13VSspec2 = corr(splits.spec13, splits.spec2,'type','spearman');
r.spec12VSspec3 = corr(splits.spec12, splits.spec3,'type','spearman');

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

% Display top-10 correlations

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