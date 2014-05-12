% Find various correlations

%% Specify directories
close all; 
clearvars -except img Feat specificity scores mem mapping;

addpath('../aux_functions');

proj_dir = '../../';
img_dir = [proj_dir '/library/cvpr_memorability_data/Data/Image data'];

%% Load data
fprintf('Loading data ... ');

if ~exist('specificity','var')
    load('../data/specificity_scores_all.mat');
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
%    load([img_dir '/target_images.mat']);
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
r.medAreavsSpec = corr(nanmedian(y(:, mapping))', specificity, ...
                       'type','spearman');
r.meanAreavsSpec = corr(nanmean(y(:, mapping))', specificity, ...
                       'type','spearman');

%for i=1:length(max_idx)
%    y(max_idx(i), mapping) = NaN; % Check carefully
%end
%second_max = nanmax(y

clear y;
                   
r.objectcountVsSpec = corr(sum(Feat.Counts(:, mapping))', specificity, ...
                     'type','spearman');
            
% Correlate object distribution with specificity

for i=1:length(mapping) % Iterate over images
    objarray = Feat.Dmemory(mapping(i)).annotation.object;
    u = 1;
    for j=1:length(objarray) % Iterate over objects
        x = objarray(j).polygon.x;
        y = objarray(j).polygon.y;
        
        if polyarea(x,y)>min_area
            geom = polygeom(x,y);
            x_cen(u) = geom(2); y_cen(u) = geom(3);
            u = u+1;
        end
        
        scatter_x(i) = std(x_cen); scatter_y(i) = std(y_cen);
        mean_x(i) = mean(x_cen); mean_y(i) = mean(y_cen);        
    end
end

r.scatterxVsSpec = corr(scatter_x',specificity,'type','spearman');
r.scatteryVsSpec = corr(scatter_y',specificity,'type','spearman');
r.meanxVsSpec = corr(mean_x',specificity,'type','spearman');
r.meanyVsSpec = corr(mean_y',specificity,'type','spearman');

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