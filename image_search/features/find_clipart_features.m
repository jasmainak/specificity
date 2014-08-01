% Save clipart features
% Abstract scene dataset can be downloaded from here:
% http://research.microsoft.com/en-us/um/people/larryz/clipart/abstract_scenes.html
%
clear all;
load('../../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');

clipart_urls = clipart_urls(51:end); % Leave out seed images

allfiles = dir('../../../library/AbstractScenes_v1/RenderedScenes/');
allfilenames = {allfiles(3:end).name};

for i=1:length(clipart_urls)
    temp = strsplit(clipart_urls{i}, '/');
    filenames{i} = cell2mat(temp(end));
end

[~, idx] = intersect(allfilenames, filenames);

% Load object occurence features
load('../../../library/AbstractScenes_v1/VisualFeatures/10K_instance_occurence_58.txt');
Feat.instance_occurence = X10K_instance_occurence_58(idx, :);

% Load co-occurence of object instances
load('../../../library/AbstractScenes_v1/VisualFeatures/10K_instance_co-occurrence_100_377.txt');
Feat.instance_cooccurence = X10K_instance_co_occurrence_100_377(idx, :);

% Load absolute location of object instances
load('../../../library/AbstractScenes_v1/VisualFeatures/10K_instance_Abs_GMM_232.txt');
Feat.instance_abslocation = X10K_instance_Abs_GMM_232(idx, :);

% Load absolute depth of object instances
load('../../../library/AbstractScenes_v1/VisualFeatures/10K_instance_Abs_depth_174.txt');
Feat.instance_absdepth = X10K_instance_Abs_depth_174(idx, :);

% Decaf
Feat.decaf = [];
for i=1:length(clipart_urls)
    load(sprintf('../../../data/image_features/decaf/clipart/%s_decaf.mat', ...
         filenames{i}), 'fc6n');
     Feat.decaf = [Feat.decaf; fc6n];
end

save('../../../data/image_features/feat_clipart.mat', 'Feat', 'clipart_urls');
