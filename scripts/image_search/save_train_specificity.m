% SAVE_TRAIN_SPECIFICITY Saves ground truth LR specificity for training
% such that none of the sentences are used in the test dataset
%
% AUTHOR: Mainak Jas
%
% See also: CALCULATE_TRAIN_SIMILARITY_INDICES,
% CALCULATE_PREDICTED_LR_SPECIFICITY

clear all; addpath('../io');
dataset = 'pascal'; overwrite = 1;

%% Load data
[scores_b, scores_w, ~, sentences, ~, url] = load_search_parameters(dataset);
[n_images, n_sentences] = size(sentences);

%% Generate sentence pair indices
comb = combntns(1:n_sentences, 2);
pairs = nchoosek(2:n_sentences-1, 2);
mask = zeros(length(comb), 1);

for i=1:size(pairs,1)
    mask = (comb(:,1)==pairs(i,1) & comb(:,2)==pairs(i,2)) | (comb(:,2)==pairs(i,1) & comb(:,1)==pairs(i,2)) | mask;
end

scores_w_clean = scores_w(:, mask);

%% For each predicted image, generate a new set of specificity values for all other images that is clean
fprintf('Finding specificity for dataset %s ... ', dataset);
for predicted_idx=1:n_images

    % Get filename of predicted image
    split_url = strsplit(url{predicted_idx}, '/');
    filename = split_url{end};
    fprintf('[%d] Predicted image = %s ... \n', predicted_idx, filename);

    load(sprintf('../../data/image_search/%s/similarity_scores/train_neg_class/%s.mat', dataset, filename), 'sample_idx');

    start_idx = find(sample_idx(1, :), 1, 'first'); end_idx = find(sample_idx(1, :), 1, 'last');
    sample_idx = sample_idx(:, start_idx:end_idx);   % trim leading and trailing zeros corresponding to ref/queries from sent1
    
    fname = sprintf('../../data/image_search/%s/LR_params/GT/predicted_img_%s.mat', dataset, filename);

    if exist(fname, 'file') && ~overwrite
        continue;
    end

    % create clean similarity scores
    scores_b_clean = zeros(size(sample_idx));
    for im_idx = 1:size(scores_b, 1)
        scores_b_clean(im_idx, :) = scores_b(im_idx, sample_idx(im_idx, :));
    end

    for idx=1:n_images

        progressbar(idx, 10, n_images);

        y_s = scores_w_clean(idx,:);
        y_d = scores_b_clean(idx,:);

        len = min(length(y_s), length(y_d));

        X = cat(2, y_s(1:len), y_d(1:len));
        labels = cat(1, ones(len,1), zeros(len,1));

        B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
    end

    save(fname, 'B');
end