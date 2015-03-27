% Seed features for clipart
% Author: Mainak Jas

clear all; close all;
load('../../../data/image_features/cvpr_2013_seed_occurence_feat.mat');

load('../../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');
filename_idx = zeros(1, length(clipart_urls));
for i=1:length(clipart_urls)
    temp = strsplit(clipart_urls{i}, '/');
    filenames{i} = cell2mat(temp(end));
    filename_idx(i) = find(strcmpi(filenames{i}, {image_objects.name}));
end

n_images = length(image_objects);
Feat.objectOccurence = zeros(n_images, 58); Feat_names.objectOccurence = labels;
Feat.type = zeros(n_images, 8); Feat_names.type = {'sky object', 'large object', 'boy', 'girl', 'animals', 'clothing', 'food', 'toys'};
Feat.x = zeros(n_images, 58);
Feat.y = zeros(n_images, 58);
Feat.z = zeros(n_images, 58);
Feat.flip = zeros(n_images, 58);
Feat.mike_pose = zeros(n_images, 7); Feat_names.mike_pose = pose;
Feat.jenny_pose = zeros(n_images, 7); Feat_names.jenny_pose = pose;
Feat.jenny_expression = zeros(n_images, 5); Feat_names.mike_expression = expression;
Feat.mike_expression = zeros(n_images, 5); Feat_names.jenny_expression = expression;

for img_idx=1:n_images
    for object_idx=1:length(image_objects(img_idx).objects)
        idx = find(strcmpi(image_objects(img_idx).objects{object_idx}, labels));
        Feat.objectOccurence(img_idx, idx) = 1;
        Feat.x(img_idx, idx) = image_objects(img_idx).x(object_idx);
        Feat.y(img_idx, idx) = image_objects(img_idx).y(object_idx);
        Feat.z(img_idx, idx) = image_objects(img_idx).z(object_idx);
        Feat.flip(img_idx, idx) = image_objects(img_idx).flip(object_idx);
        Feat.type(img_idx, image_objects(img_idx).type(object_idx) + 1) = 1;
    end
    if ~isempty(image_objects(img_idx).mike_exp)
        idx = find(strcmpi(image_objects(img_idx).mike_exp, expression));
        Feat.mike_expression(img_idx, idx) = 1;
    end
    if ~isempty(image_objects(img_idx).jenny_exp)
        idx = find(strcmpi(image_objects(img_idx).jenny_exp, expression));
        Feat.jenny_expression(img_idx, idx) = 1;
    end
    if ~isempty(image_objects(img_idx).mike_pose)
        idx = find(strcmpi(image_objects(img_idx).mike_pose, pose));
        Feat.mike_pose(img_idx, idx) = 1;
    end
    if ~isempty(image_objects(img_idx).jenny_pose)
        idx = find(strcmpi(image_objects(img_idx).jenny_pose, pose));
        Feat.jenny_pose(img_idx, idx) = 1;
    end
end

% find objects that occur at least 100 times
idx = find(sum(Feat.objectOccurence, 1) >= 100);

% Find co-occurence matrices
u = 1;
for i=1:length(idx)
    for j=i+1:length(idx)
        Feat_names.objectCooccurence(u) = strcat(labels(idx(i)), '-', labels(idx(j)));
        u = u + 1;
    end
end
Feat.objectCooccurence = zeros(n_images, length(Feat_names.objectCooccurence));

for img_idx=1:n_images
    u = 1;
    for i=1:length(idx)
        for j=i+1:length(idx)
            if Feat.objectOccurence(img_idx, idx(i)) == 1 && Feat.objectOccurence(img_idx, idx(j)) == 1
                Feat.objectCooccurence(img_idx, u) = 1;
            end
            u = u + 1;
        end
    end
end

% Truncate features for only 500 images
feat_names = fieldnames(Feat);
for i=1:length(feat_names)
    feat = Feat.(feat_names{i});
    Feat.(feat_names{i}) = feat(filename_idx, :);
end

fprintf('Saving features ... \n');
save('../../../data/image_features/feat_clipart.mat', 'Feat', 'Feat_names', 'clipart_urls');
