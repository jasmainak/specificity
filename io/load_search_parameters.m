function [scores_b, scores_w, s, sentences, m_sentences, url, sent_pairs] = load_search_parameters(dataset, request_sentpairs)

addpath('../aux_functions');
fprintf('\nLoading %s dataset ... ', dataset);

if nargin < 2
    request_sentpairs = 0;
end

parameter_fname = ['../../data/search_parameters/search_parameters_' dataset '.mat'];
if exist(parameter_fname, 'file')
    fprintf('from file %s', parameter_fname);
    load(parameter_fname, 'scores_b', 'scores_w', 's', 'sentences', ...
         'm_sentences', 'url');  % Do not load sent_pairs as it takes a long time to load
    fprintf('[Done] \n');
    
    if request_sentpairs
        fprintf('loading sent pairs ... ');
        load(parameter_fname, 'sent_pairs');
        fprintf('[Done]\n');
    end

    return;
end

% LOAD DATASET SEARCH PARAMETERS
if strcmpi(dataset, 'pascal')

    load('../../data/image_search_50sentences_query.mat');
    load('../../data/image_search_50sentences_parameters.mat');

    load('../../data/sentences/pascal_1000_img_50_sent.mat', 'pascal_urls');
    url = pascal_urls;

    scores_b = []; sent_pairs = [];
    for i=1:length(pascal_urls)
        progressbar(i, 10, 1000);
        split_url = strsplit(pascal_urls{i}, '/');
        filename = [split_url{end} '.mat'];
        X = load(sprintf('../../data/search_parameters/pascal/mu_d/img_%s', filename), 'scores_b', 'sent_pairs');
        scores_b = cat(1, scores_b, X.scores_b);
        sent_pairs = cat(3, sent_pairs, X.sent_pairs);
    end

    m_sentences = 40; sent_pairs = permute(sent_pairs, [3 1 2]);
    save(parameter_fname, 'scores_b', 'scores_w', 's', 'sentences', ...
         'm_sentences', 'url', 'sent_pairs');

elseif strcmpi(dataset, 'clipart')

   X = load('../../data/sentences/clipart_500_img_48_sent.mat');
   sentences = X.clipart_sentences;

   load('../../data/search_parameters/clipart/image_search_mud.mat');

   load('../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');
   url = clipart_urls;

   scores_w = []; scores_b = []; s = []; sent_pairs = [];
   for i=0:curr_idx

       progressbar(i, 10, curr_idx + 1);
       split_url = strsplit(clipart_urls{i + 1}, '/');
       filename = [split_url{end} '.mat'];

       X = load(sprintf('../../data/search_parameters/clipart/mus/image_search_mus_%d.mat',i));
       scores_w = cat(1, scores_w, X.scores_w);

       X = load(sprintf('../../data/search_parameters/clipart/mu_d/img_%s', filename), 'scores_b', 'sent_pairs');
       scores_b = cat(1, scores_b, X.scores_b);
       sent_pairs = cat(3, sent_pairs, X.sent_pairs);

       X = load(sprintf('../../data/search_parameters/clipart/s/image_search_s_%d.mat', i));
       s = cat(1, s, X.s);

   end

   m_sentences = 40; sent_pairs = permute(sent_pairs, [3 1 2]);

   save(parameter_fname, 'scores_b', 'scores_w', 's', 'sentences', ...
         'm_sentences', 'url', 'sent_pairs');

end

fprintf(' [Done]\n');

end