clear all; close all;

n_scenes = 10020; % 1002 or 10020
overwrite = 0; 
input_file = 'Scenes_10020.txt'; %'SeedScenes_1002.txt';

%%%%% Create features from Larry's file %%%%%
if ~exist('iccv13/Attributes_9_24_3124.txt') || overwrite
    cd('iccv13/');
    system(sprintf('./a.out %s 0 0 0 %d', input_file, n_scenes));
    cd('..');
end

%%%%% Number of features / starting index of features etc. %%%%%
NUM_GM_SPAT = 9; NUM_GM_REL = 24;

n_features.n_objectinstances = 58;

n_features.depth = 3; n_features.poses = 7; n_features.expressions = 5;
n_features.clothing = 10; n_features.person_attributes = n_features.poses + n_features.clothing + n_features.expressions;

n_features.all = 1 + NUM_GM_SPAT*n_features.depth + n_features.n_objectinstances + ...
    NUM_GM_REL*n_features.n_objectinstances + NUM_GM_REL*n_features.n_objectinstances + ...
    3*n_features.n_objectinstances + n_features.person_attributes + n_features.n_objectinstances;

m.AbsoluteSpatialIdx = 2; % Add one because it is in MATLAB
m.CoOccurrenceIdx = m.AbsoluteSpatialIdx + NUM_GM_SPAT*n_features.depth;
m.RelativeSpatialIdx = m.CoOccurrenceIdx + n_features.n_objectinstances;
m.RelativeSpatialFlipIdx = m.RelativeSpatialIdx + NUM_GM_REL*n_features.n_objectinstances;
m.RelativeDepthIdx = m.RelativeSpatialFlipIdx + NUM_GM_REL*n_features.n_objectinstances;
m.ExpressionIdx = m.RelativeDepthIdx + 3*n_features.n_objectinstances;
m.PoseIdx = m.ExpressionIdx + n_features.expressions;
m.ClothingIdx = m.PoseIdx + n_features.poses;
m.HandIdx = m.ClothingIdx + n_features.clothing;

%%%%% Features for each scene %%%%%

feat_unformatted = dlmread('iccv13/Attributes_9_24_3124.txt');

m_features = 3125;
idx_scene = 1;
num_clipart = zeros(n_scenes,1);
feat_scenes = cell(n_scenes, 1);

for idx_row=1:size(feat_unformatted,1);
    
    % either read number of clipart objects in scene
    if length(find(feat_unformatted(idx_row, :) > 0)) == 1
        if idx_scene > 1
            feat_scenes{idx_scene - 1} = feat_scene;
        end
        num_clipart(idx_scene) = feat_unformatted(idx_row, 1);
        feat_scene = zeros(num_clipart(idx_scene), m_features);
        idx_scene = idx_scene + 1; idx_clipart = 1;
    % or read the features from each clipart object
    else
        feat_scene(idx_clipart, :) = feat_unformatted(idx_row, :);
        idx_clipart = idx_clipart + 1;
    end
end

feat_scenes{idx_scene - 1} = feat_scene;

%%%%% Extract attributes in each scene %%%%%

Feat.ObjectOccurence = zeros(n_scenes, n_features.n_objectinstances);
Feat.person = zeros(n_scenes, n_features.person_attributes);

for i=1:length(feat_scenes)
    feat_scene = feat_scenes{i};
    if isempty(feat_scene)
        continue;
    end
    Feat.person(i, :) = sum(feat_scene(:, m.ExpressionIdx + 1: m.ExpressionIdx + n_features.person_attributes), 1) > 0;
    for j=1:size(feat_scenes{i}, 1)
        feat_scene = feat_scenes{i};
        Feat.ObjectOccurence(i, feat_scene(j, 1) + 1) = 1;  % Add one because it is MATLAB
        
    end
end