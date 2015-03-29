% LOAD_SEARCH_PARAMETERS
%
% Author: Mainak Jas
%
% Load the search parameters: 
% scores_w: 
% scores_b: LR
% sentences:
% url:
% sent_pairs

function [scores_b, scores_w, s, sentences, m_sentences, urls, sent_pairs] = load_search_parameters(dataset, request_sentpairs)

addpath('../aux_functions');
fprintf('\nLoading %s dataset ... ', dataset);

if nargin < 2
    request_sentpairs = 0;
end

if strcmpi(dataset, 'clipart')
    X = load('../../data/sentences/clipart_500_img_48_sent.mat');
    sentences = X.clipart_sentences;
    urls = X.clipart_urls;
elseif strcmpi(dataset, 'pascal')
    X = load('../../data/sentences/pascal_1000_img_50_sent.mat');
    sentences = X.pascal_sentences;
    urls = X.pascal_urls;
end

parameters_dir = sprintf('../../data/image_search/%s/similarity_scores', dataset);
n_images = length(urls);

scores_w = []; scores_b = []; s = []; sent_pairs = [];
for i=1:n_images
    progressbar(i-1, 10, n_images);
    split_url = strsplit(urls{i}, '/');
    filename = [split_url{end} '.mat'];
    
    X = load(sprintf('%s/train_pos_class/%s', parameters_dir, filename));
    scores_w = cat(1, scores_w, X.scores_w);
    
    X = load(sprintf('%s/train_candidate_neg_class/%s', parameters_dir, filename), 'scores_b', 'sent_pairs');
    scores_b = cat(1, scores_b, X.scores_b);
    
    if request_sentpairs
        sent_pairs = cat(3, sent_pairs, X.sent_pairs);
    end
    
    X = load(sprintf('%s/test/%s', parameters_dir, filename));
    s = cat(1, s, X.s);
end

m_sentences = 40; sent_pairs = permute(sent_pairs, [3 1 2]);

fprintf(' [Done]\n');

end