clearvars -except img; close all;

if ~exist('img','var')
    load('../library/cvpr_memorability_data/Data/Image data/target_images.mat');
end

load('../data/specificity_scores_all.mat');
load('../data/memorability_mapping.mat');

for i=1:length(specificity)
    I = img(:,:,:,mapping(i));
    
    r = I(:,:,1)./255; 
    g = I(:,:,2)./255; 
    b = I(:,:,3)./255;
    
    rs(i) = mean(r(:)); gs(i) = mean(g(:)); bs(i) = mean(b(:));
    [h(i), s(i), v(i)] = rgb2hsv([rs(i), gs(i), bs(i)]);
end

rho.r = corr(rs', specificity, 'type', 'spearman'); % correlation with red
rho.g = corr(gs', specificity, 'type', 'spearman'); % correlation with green
rho.b = corr(bs', specificity, 'type', 'spearman'); % correlation with blue
rho.h = corr(h', specificity, 'type', 'spearman'); % correlation with hue
rho.s = corr(s', specificity, 'type', 'spearman'); % correlate with saturatino
rho.v = corr(v', specificity, 'type', 'spearman'); % correlate with intensity