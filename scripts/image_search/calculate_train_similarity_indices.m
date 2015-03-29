% CALCULATE_TRAIN_SIMILARITY_INDICES finds the subset of the computed
% candidate similarities for training
%
% INPUT
%   data/image_search/{dataset}/similarity_scores/*
%
% OUTPUT
%   data/image_search/{dataset}/similarity_scores/train_neg_class/*
%
% AUTHOR: Mainak Jas
%
% See also: save_train_specificity
function calculate_train_similarity_indices(dataset)

addpath('../io/');
[scores_b, ~, ~, sentences, m_sentences, urls, sent_pairs] = load_search_parameters(dataset, 1);

if strcmpi(dataset, 'pascal')
    k_sentences = 24; % number of sentences m_sentences must be reduced to
elseif strcmpi(dataset, 'clipart')
    k_sentences = 23;
end

img_idxs = cell2mat(squeeze(sent_pairs(:, :, 3)));
other_sents = squeeze(sent_pairs(:, :, 4));
sent_idxs = zeros(size(img_idxs));
n_sentences = size(sentences, 2);
h = waitbar(0, 'Calculating sentence idx ...');
for i=1:size(img_idxs, 1)
    waitbar(i/size(img_idxs,1));
    for j=1:size(img_idxs, 2)
        sent_idx = strmatch(other_sents{i, j}, sentences(img_idxs(i,j)+1, :));
        
        % if it's query/reference then put a red flag
        if ~isempty(find(sent_idx == 0 | sent_idx == n_sentences - 1))
            sent_idxs(i, j) = 0;
        else   % just pick one of the sentence indices if it's not query/reference
            sent_idxs(i, j) = sent_idx(1);
        end
    end
end
close(h);

for predicted_idx = 0:size(scores_b,1) - 1
    split_url = strsplit(urls{predicted_idx + 1}, '/');
    filename = [split_url{end} '.mat'];    
    fprintf('[%d] %s ... ', predicted_idx, filename);

    sample_idx = clean_predicted_idx(predicted_idx, scores_b, sentences, m_sentences, sent_pairs, sent_idxs, k_sentences);
    fprintf('Saving ... ');
    save(sprintf('../../data/image_search/%s/similarity_scores/train_neg_class/%s', dataset, filename), ...
         'sample_idx', 'predicted_idx', '-v7.3');
    fprintf('[Done]\n');
end

end

function sample_idx = clean_predicted_idx(predicted_idx, scores_b, sentences, m_sentences, sent_pairs, sent_idxs, k_sentences)
img_idxs = cell2mat(squeeze(sent_pairs(:, :, 3)));

[n_images, n_sentences] = size(sentences);

sample_idx = zeros(n_images, k_sentences*n_sentences);
for im_idx = 1:size(scores_b,1)
    for sent_idx = 2:n_sentences-1   % loop through sentences in image1 of the pair leaving out reference and query sentences
        start = (sent_idx - 1)*m_sentences + 1;
        stop = sent_idx*m_sentences;
        chunk_idx = start:stop;

        % remove pairs (s1, s2) where s2 belongs to image predicted_idx
        contaminated_img_idx = find(img_idxs(im_idx, start:stop) == predicted_idx);
        % remove pairs (s1, s2) where s2 is ref/query sentence
        contaminated_sent_idx = find(sent_idxs(im_idx, start:stop) == 0 | sent_idxs(im_idx, start:stop) == n_sentences - 1);
        
        contaminated_idx = [contaminated_img_idx, contaminated_sent_idx];
        
        clean_idx = setdiff(1:m_sentences, contaminated_idx);
        
        start = (sent_idx - 1)*k_sentences + 1;
        stop = sent_idx*k_sentences;

        sample_idx(im_idx, start:stop) = sort(randsample(chunk_idx(clean_idx), k_sentences));
    end
end

end