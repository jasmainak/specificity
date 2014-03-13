% Find correlationss with attributes

close all; clear all;

load('../library/annotations/annotations/anno_feats.mat');
load('../library/annotations/annotations/anno_names.mat');
load('../data/specificity_scores.mat');
load('../data/memorability_mapping.mat');

for i=1:length(anno_names)
    r(i) = corr(anno_feats(mapping, i), specificity, 'type', 'spearman');
end

% sort and clip correlation values
r(isnan(r)) = -999;
[r_sorted, idx] = sort(r, 'descend');
r_sorted = r_sorted(r_sorted~=-999); 

for i=1:40
    fprintf('%f %s\n', r_sorted(i), anno_names{idx(i)});
end

% For better display while plotting
anno_names = strrep(anno_names, '_', '\_'); 

last_r = r_sorted(1);
plot_interval = 0.015;
text(double(1), double(r_sorted(1)), anno_names{idx(1)});

for i=1:length(r_sorted)
    
    if r_sorted(i) < last_r - plot_interval
        last_r = r_sorted(i);
        text(double(i), double(r_sorted(i)), anno_names{idx(i)});
    end
end

set(gca,'XLim',[0 1000], 'YLim', [-0.4 0.4], 'XTick',[],'Tickdir','out');
title('Correlating attribute + scene annotations with specificity');