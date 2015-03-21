clear all;
addpath('../io/');
addpath('utils/');

load('../../data/search_parameters/clipart/cosine/gt_LR/predicted_img_Scene873_0.png.mat');

[scores_b, scores_w, s, sentences, ~, url] = load_search_parameters('clipart_cosine');
% [n_images, n_sentences] = size(sentences);
% 
% comb = combntns(1:n_sentences, 2);
% pairs = nchoosek(2:n_sentences-1, 2);
% mask = zeros(length(comb), 1);
% 
% for i=1:size(pairs,1)
%     mask = (comb(:,1)==pairs(i,1) & comb(:,2)==pairs(i,2)) | (comb(:,2)==pairs(i,1) & comb(:,1)==pairs(i,2)) | mask;
% end
% 
% scores_w_clean = scores_w(:, mask);
% 
% n_images = length(url);
% 
% for idx=1:n_images
% 
%     progressbar(idx, 10, n_images);
% 
%     %sent_pair = sent_pairs(idx, :, :);
% 
%     y_s = scores_w_clean(idx,:);
%     y_d = scores_b(idx,:);
% 
%     X = cat(2, y_s, y_d);
%     labels = cat(1, ones(length(y_s),1), zeros(length(y_d),1));
% 
%     B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
% end

y = B(:, 1); z = B(:, 2);

rank_b = baseline_search(s);
rank_s = specificity_search(s, y, z);