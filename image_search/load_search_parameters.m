function [scores_b, scores_w, s, sentences, m_sentences, url] = load_search_parameters(dataset)

addpath('../aux_functions');
fprintf('\nLoading %s dataset ... ', dataset);

% LOAD DATASET SEARCH PARAMETERS
if strcmpi(dataset, 'pascal')
    load('../../data/image_search_50sentences_query.mat');
    load('../../data/image_search_50sentences_parameters.mat');

    scores_b = [];
    for i=1:1000
        progressbar(i, 10, 1000);
        X = load(sprintf('../../data/search_parameters/pascal/mu_d/image_search_50sentences_mud_%d.mat', i - 1), 'scores_b');
        scores_b = cat(1, scores_b, X.scores_b);
    end

    m_sentences = 24;

    load('../../data/sentences/pascal_1000_img_50_sent.mat', 'pascal_urls');
    url = pascal_urls;

elseif strcmpi(dataset, 'clipart')

   X = load('../../data/sentences/clipart_500_img_48_sent.mat');
   sentences = X.clipart_sentences;

   load('../../data/search_parameters/clipart/image_search_mud.mat');

   scores_w = []; scores_b = []; s = [];
   for i=0:curr_idx

       progressbar(i, 10, curr_idx + 1);

       X = load(sprintf('../../data/search_parameters/clipart/mus/image_search_mus_%d.mat',i));
       scores_w = cat(1, scores_w, X.scores_w);

       X = load(sprintf('../../data/search_parameters/clipart/mu_d/image_search_50sentences_mud_%d.mat',i));
       scores_b = cat(1, scores_b, X.scores_b);

       X = load(sprintf('../../data/search_parameters/clipart/s/image_search_s_%d.mat', i));
       s = cat(1, s, X.s);

   end

   m_sentences = 23;

   load('../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');
   url = clipart_urls;

end

fprintf(' [Done]');

end