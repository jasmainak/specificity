% Find correlationss with attributes

close all; clear all;

addpath('../../library/rotateXLabels/');
addpath('../../library/export_fig/');

load('../../library/annotations/annotations/anno_feats_modified.mat');
load('../../library/annotations/annotations/anno_names.mat');
load('../../data/specificity_scores_all.mat');
load('../../data/memorability_mapping.mat'); 
load('../../library/cvpr_memorability_data/Data/Image data/target_images.mat');

% Remove all features after the scenes
anno_feats = anno_feats(:, 1:789);
anno_names = anno_names(:, 1:789);

mapping = mapping(1:length(specificity));
img_min = img(:, :, :, mapping);
anno_feats_min = anno_feats(mapping, :);

%exclude_attributes = {'TL_building', 'BL_building', 'BR_building', 'TR_building', ...
%                      'clear_sky', 'area_building', 'count_building', 'count_ground', ...
%                      'area_sky', 'TL_person', 'BL_person', 'BR_person', 'TR_person', ...
%                      'hair_brown', 'hair_black', 'hair_short', 'blue_sky', 'age_teen', ...
%                      'age_adult', 'area_ground', 'count_sky', 'shorts', 'gas', 'race_C', ...
%                      'attire_cas', 't-shirt'};

exclude_attributes = {'BL_building', 'TR_building', 'TL_building', 'area_building', 'count_building', 'building', ...
                      'BL_person', 'BR_person', 'TL_person', 'TR_person', 'race_C', 'hair_brown', 'hair_short', 'hair_black', 'attire_cas', 'shorts', 'age_teen', 'age_adult', 't-shirt', ...
                      'area_sky', 'blue_sky', 'sky_present', 'count_sky', 'sky', ...
                      'ground', 'area_ground', ...
                      'window', 'area_window', ...
                      'count_plant', ...
                      'cultural-or-historical-place'};

for i=1:length(exclude_attributes)
    idx = strmatch(exclude_attributes{i}, anno_names);
    anno_names = anno_names([1:idx-1, idx+1:end]);
    anno_feats = anno_feats(:, [1:idx-1, idx+1:end]);
    anno_feats_min = anno_feats_min(:, [1:idx-1, idx+1:end]);
end

%anno_name = 'empty_space';
%fp = fopen(['~/Desktop/' anno_name '.html'], 'w');
%idx = strmatch(anno_name, anno_names);
%positive_imgs = find(anno_feats_min(:, idx) > 0);
%for i=1:length(positive_imgs)
%    fprintf(fp, '<img src=''http://neuro.hut.fi/~mainak/sampled_images/img_%d.jpg''></img><br/>\n', positive_imgs(i));
%    fprintf(fp, '%s = %0.4f<br/>\n', anno_name, anno_feats_min(positive_imgs(i), idx));
%end
%fclose(fp);

old_names = {'BR_building', 'area_plant', 'clear_sky', 'count_window', 'count_ground'};
new_names = {'building', 'plant', 'sky', 'window', 'ground'};
%old_names = {'representation', 'sky_present', 'face_visible', 'race_C', 'is_strange', 'recognize_place', 'attire_cas'};
%new_names = {'presentation, display', 'is sky present?', 'is face visible?', 'race is Caucasian', 'is strange?', 'recognize place?', 'casual attire'};

for i=1:length(old_names)
    anno_names = strrep(anno_names, old_names{i}, new_names{i});
end

plot = 'False';
expt = 'specificity'; % {'abs_diff', 'specificity'}

for i=1:length(anno_names)
    r_specificity(i) = corr(anno_feats(mapping, i), specificity, 'type', 'spearman');
    r_memorability(i) = corr(anno_feats(mapping, i), mem(mapping), 'type', 'spearman');
    %r_memorability(i) = corr(anno_feats(:, i), mem, 'type', 'spearman');
end

if strcmpi(expt, 'abs_diff')
    
    % filter results to include only abs(correlation) > 0.1
    include = find(abs(r_memorability)>0.1 & abs(r_specificity)>0.1);
    
    r = abs(r_memorability - r_specificity);
    
    r = r(include); 
    r_memorability = r_memorability(include);    
    r_specificity = r_specificity(include);
    anno_names_min = anno_names(include);     
    
    [r_sorted, idx] = sort(r, 'descend');
            
    fprintf('Specificity\tMemorability\tDifference\tAttribute\n');
    
    for i=1:40
        fprintf('%0.3f\t\t   %0.3f   \t%0.3f   \t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), r_sorted(i), anno_names_min{idx(i)});
    end
    break;
end

if strcmpi(expt, 'memorability')
   
    include = find(~isnan(r_memorability));
    r_specificity = r_specificity(include); r_memorability = r_memorability(include);
    anno_names_min = anno_names(include);
    
    [r_sorted, idx] = sort(r_memorability, 'descend');
    fprintf('Mem\tSpec\tAttribute\n');
    
    for i=1:40
        fprintf('%0.3f\t%0.3f\t%s\n', r_memorability(idx(i)), r_specificity(idx(i)), ...
               anno_names_min{idx(i)});
    end
    
    fprintf('\n');
    
    fprintf('Mem\tSpec\tAttribute\n');
    
    for i=length(r_sorted):-1:length(r_sorted)-39
        fprintf('%0.3f\t%0.3f\t%s\n', r_memorability(idx(i)), r_specificity(idx(i)), ...
                anno_names_min{idx(i)});
    end
    
    break;
end

if strcmpi(expt, 'specificity')
    
    % Filter results to discard NaNs
    include = find(~isnan(r_specificity));
    r_specificity = r_specificity(include); r_memorability = r_memorability(include);
    anno_names_min = anno_names(include);
    
    [r_sorted, idx] = sort(r_specificity, 'descend');
    
    
    fprintf('Spec\tMem\tAttribute\n');
    
    for i=1:40
        fprintf('%0.3f\t%0.3f\t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), ...
               anno_names_min{idx(i)});
    end
    
    fprintf('\n');
    
    fprintf('Spec\tMem\tAttribute\n');
    
    for i=length(r_sorted):-1:length(r_sorted)-39
        fprintf('%0.3f\t%0.3f\t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), ...
                anno_names_min{idx(i)});
    end
    
    anno_names_min = strrep(anno_names_min, '_', ' ');
    
    bar([r_specificity(idx(end:-1:end-9)), r_specificity(idx(1:10))]);
    set(gca, 'XTick', 1:20, 'XTickLabel', ...
        [anno_names_min(idx(end:-1:end-9)), anno_names_min(idx(1:10))], ...
        'XLim', [0 21], 'Fontsize', 12, 'TickLength', [0.007, 0.007]);
    rotateXLabels(gca, 90); grid on;
    ylabel('Spearman''s \rho', 'Fontsize', 12);
end

export_fig '../../plots/paper/attribute_correlations.pdf' -transparent;