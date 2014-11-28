% Save clipart features
% Abstract scene dataset can be downloaded from here:
% http://research.microsoft.com/en-us/um/people/larryz/clipart/abstract_scenes.html
%
clear all;
features_dir = '../../../library/AbstractScenes_v1/VisualFeatures/';
load('../../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');

% clipart_urls = clipart_urls(51:end); % Don't leave out seed images

for i=1:length(clipart_urls)
    temp = strsplit(clipart_urls{i}, '/');
    filenames{i} = cell2mat(temp(end));
end

for i=1:length(filenames)
    str_idx = strsplit(filenames{i}(6:end-4), '_');
    idx(i) = str2num(str_idx{1})*10 + str2num(str_idx{2}) + 1;
end

features = {'10K_instance_occurence_58', '10K_category_occurence_11', '10K_instance_Abs_GMM_232', ...
            '10K_instance_Abs_depth_174', '10K_category_Abs_GMM_44', ...
            '10K_person_24', '10K_instance_hand_116', ...
            '10K_instance_head_116', '10K_category_Rel_GMM_100_264', ...
            '10K_instance_Rel_GMM_100_1508', '10K_instance_Rel_depth_100_1131'};
fdnames = {'instance_occurence', 'category_occurence', 'instance_cooccurence', ...
           'instance_absdepth', 'instance_abslocation', ...
           'person', 'instance_hand', ...
           'instance_head', 'category_relocation', ...
           'instance_relocation', 'instance_reldepth'};
        
% Load features
Feat = struct([]); Feat_names = struct([]);
for i=1:length(features)
    fprintf(['Loading ' fdnames{i} '... \n']);
    feat = load([features_dir features{i} '.txt']);
    Feat(1).(fdnames{i}) = feat(idx, :);
    feat_name = textread([features_dir features{i} '_names.txt'], '%s%*[^\n]');
    Feat_names(1).(fdnames{i}) = feat_name;
end

% Decaf
Feat.decaf = [];
for i=1:length(clipart_urls)
    load(sprintf('../../../data/image_features/decaf/clipart/%s_decaf.mat', ...
         filenames{i}), 'fc6n');
     Feat.decaf = [Feat.decaf; fc6n];
end

fprintf('Saving features ... \n');
save('../../../data/image_features/feat_clipart.mat', 'Feat', 'Feat_names', 'clipart_urls');
