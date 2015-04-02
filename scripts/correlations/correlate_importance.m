% CORRELATE_IMPORTANCE computes the correlation of specificity with
% importance.
%
% AUTHOR: Mainak Jas

clear all; close all;

%% Load data
load('../../data/sentences/memorability_888_img_5_sent.mat');
load('../../data/memorability_mapping.mat');
load('../../data/specificity_scores_all.mat');
load('../../data/object_presence.mat');
load('../../data/importance_scores.mat');

%% Remove 'person sitting' category
object_pres = object_pres([1:22, 24:41], :);
objectnames = objectnames([1:22, 24:41]);
importance = importance([1:22, 24:41]);
S = S([1:22, 24:41]); D = D([1:22, 24:41]);

%% Correlation of importance with mean specificity of that object
for i=1:length(importance)    
    im_idx = find(object_pres(i, :)==1);
    obj_score(i) = mean(specificity(im_idx));
end

fprintf('Correlation of importance with specificity = %0.2f\n', ...
        corr(obj_score', importance, 'type', 'spearman'));