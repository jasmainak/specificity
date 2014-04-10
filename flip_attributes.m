close all; clear all;

load('../library/annotations/annotations/anno_feats.mat');
load('../library/annotations/annotations/anno_names.mat');

attr = {'enclosed_space', 'perspective_view', 'empty_space', ...
        'mirror_symmetry', 'pleasant_scene', 'unusual_scene', 'dull_colors', ...
        'expert_photography', 'attractive', 'memorable', 'central_object',...
        'single_focus', 'zoomed_in', 'top_down', 'sky_present', 'clear_sky', ...
        'blue_sky'};
    
for i=1:length(attr)
    idx(i) = find(strcmpi(anno_names, attr{i}));
end

anno_feats(:, idx) = 1 - anno_feats(:, idx);

% Make clear_sky and blue_sky zero when sky_present < 0.5

sky_idx = find(anno_feats(:, strcmpi(anno_names, 'sky_present')) < 0.5);

anno_feats(sky_idx, strcmpi(anno_names, 'clear_sky')) = 0;
anno_feats(sky_idx, strcmpi(anno_names, 'blue_sky')) = 0;

save('../library/annotations/annotations/anno_feats_modified.mat', 'anno_feats');