clear all;
dataset = 'pascal';

[scores_b, scores_w, ~, sentences, ~, url, sent_pairs] = load_search_parameters(dataset);
[n_images, ~] = size(sentences);

fprintf('Finding specificity for dataset %s ... ', dataset);

for predicted_idx=1:n_images

    split_url = strsplit(url{predicted_idx}, '/');
    filename = split_url{end};
    fprintf('[%d] Predicted image = %s ... ', predicted_idx, filename);
    load(sprintf('../../data/search_parameters/%s/mu_d_cleaned/mu_d_%d.mat', dataset, predicted_idx - 1), 'sample_idx');

    fname = sprintf('../../data/search_parameters/%s/LR/predicted_img_%s.mat', dataset, filename);

    if exist(fname, 'file')
        continue;
    end

    % create clean similarity scores
    scores_b_clean = zeros(size(sample_idx));
    for im_idx = 1:size(scores_b, 1)
        scores_b_clean(im_idx, :) = scores_b(im_idx, sample_idx(im_idx, :));
    end

    for idx=1:n_images

        progressbar(idx, 10, n_images);

        sent_pair = sent_pairs(idx, :, :);

        y_s = scores_w(idx,:);
        y_d = scores_b_clean(idx,:);

        len = min(length(y_s), length(y_d));

        X = cat(2, y_s(1:len), y_d(1:len));
        labels = cat(1, ones(len,1), zeros(len,1));

        B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
    end

    save(fname, 'B');
end