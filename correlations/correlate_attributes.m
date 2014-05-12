% Find correlationss with attributes

close all; clear all;

load('../../library/annotations/annotations/anno_feats_modified.mat');
load('../../library/annotations/annotations/anno_names.mat');
load('../../data/specificity_scores_all.mat');
load('../../data/memorability_mapping.mat'); 

mapping = mapping(1:length(specificity));

plot = 'False';
expt = 'abs_diff'; % {'abs_diff', 'specificity'}

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
    anno_names = anno_names(include);     
    
    [r_sorted, idx] = sort(r, 'descend');
            
    fprintf('Specificity\tMemorability\tDifference\tAttribute\n');
    
    for i=1:20
        fprintf('%0.3f\t\t   %0.3f   \t%0.3f   \t%s\n', r_specificity(idx(i)), r_memorability(idx(i)), r_sorted(i), anno_names{idx(i)});
    end
    
end

if strcmpi(expt, 'specificity')
    
    % Filter results to discard NaNs
    include = find(~isnan(r_specificity));
    r_specificity = r_specificity(include);
    
    [r_sorted, idx] = sort(r_specificity, 'descend');
        
    for i=1:10
        fprintf('%0.3f\t%s\n', r_specificity(idx(i)), anno_names{idx(i)});
    end
    
    fprintf('\n');
    
    for i=length(r_sorted):-1:length(r_sorted)-9
        fprintf('%0.3f\t%s\n', r_specificity(idx(i)), anno_names{idx(i)});
    end
end