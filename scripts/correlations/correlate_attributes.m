% Author: Mainak Jas
% Find correlationss with attributes

close all; clear all;

addpath('../../library/rotateXLabels/');
addpath('../../library/export_fig/');

%% Load data and preprocess
load('../../library/annotations/annotations/anno_feats_modified.mat');
load('../../library/annotations/annotations/anno_names.mat');
load('../../data/specificity_scores_all.mat');
load('../../data/memorability_mapping.mat'); 
load('../../data/target_images.mat');

% Remove all features after the scenes
anno_feats = anno_feats(:, 1:789);
anno_names = anno_names(:, 1:789);

mapping = mapping(1:length(specificity));
img_min = img(:, :, :, mapping);
anno_feats_min = anno_feats(mapping, :);

% These attributes are basically manifestations of the attributes in top-10
% Hence they are excluded to avoid repetition
exclude_attributes = {'BL_building', 'TR_building', 'TL_building', 'area_building', 'count_building', 'building', ...
                      'BL_person', 'BR_person', 'TL_person', 'TR_person', 'race_C', 'hair_brown', 'hair_short', 'hair_black', 'attire_cas', 'shorts', 'age_teen', 'age_adult', 't-shirt', ...
                      'area_sky', 'blue_sky', 'sky_present', 'count_sky', 'sky', ...
                      'ground', 'area_ground', 'window', 'area_window', ...
                      'count_plant', 'cultural-or-historical-place'};
for i=1:length(exclude_attributes)
    idx = strmatch(exclude_attributes{i}, anno_names);
    anno_names = anno_names([1:idx-1, idx+1:end]);
    anno_feats = anno_feats(:, [1:idx-1, idx+1:end]);
    anno_feats_min = anno_feats_min(:, [1:idx-1, idx+1:end]);
end

% Rename some attributes to be human-readable
old_names = {'BR_building', 'area_plant', 'clear_sky', 'count_window', 'count_ground', 'a/airport_terminal'};
new_names = {'building', 'plant', 'sky', 'window', 'ground', 'airport terminal'};

for i=1:length(old_names)
    anno_names = strrep(anno_names, old_names{i}, new_names{i});
end

%% Compute Correlations
for i=1:length(anno_names)
    r_specificity(i) = corr(anno_feats(mapping, i), specificity, 'type', 'spearman');
    r_memorability(i) = corr(anno_feats(mapping, i), mem(mapping), 'type', 'spearman');
end

% Filter results to discard NaNs
include = find(~isnan(r_specificity));
r_specificity = r_specificity(include); r_memorability = r_memorability(include);
anno_names_min = anno_names(include);

% TOP-10 results
[r_sorted, idx] = sort(r_specificity, 'descend');
N = 10;
fprintf('Spec\tMem\tAttribute\n');
for i=1:N
    fprintf('%0.3f\t%0.3f\t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), ...
        anno_names_min{idx(i)});
end
fprintf('\n');

% BOTTOM-10 results
fprintf('Spec\tMem\tAttribute\n');
for i=length(r_sorted):-1:length(r_sorted)-N+1
    fprintf('%0.3f\t%0.3f\t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), ...
        anno_names_min{idx(i)});
end

% peaceful, painting, postcard
attributes = {'peaceful', 'buy_painting', 'on_post-card'};
for i=1:length(attributes)
    attribute_idx = find(strcmpi(attributes{i}, anno_names_min));
    fprintf('\nCorrelation of specificity and %s is %0.2f\n', attributes{i}, r_specificity(attribute_idx));
    fprintf('Correlation of memorability and %s is %0.2f\n', attributes{i}, r_memorability(attribute_idx));
end

%% Make Figure
anno_names_min = strrep(anno_names_min, '_', ' ');

bar([r_specificity(idx(end:-1:end-9)), r_specificity(idx(1:10))]);
set(gca, 'XTick', 1:20, 'XTickLabel', ...
    [anno_names_min(idx(end:-1:end-9)), anno_names_min(idx(1:10))], ...
    'XLim', [0 21], 'Fontsize', 12, 'TickLength', [0.007, 0.007]);
rotateXLabels(gca, 45); grid on;
ylabel('Spearman''s \rho', 'Fontsize', 12);

export_fig '../../plots/paper/attribute_correlations.pdf' -transparent;