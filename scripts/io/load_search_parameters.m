% Author: Mainak Jas
%
% Load the search parameters: 
% scores_w: 
% scores_b: LR
% sentences:
% url:
% sent_pairs

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

if strcmpi(dataset, 'clipart_cosine')
    X = load('../../data/sentences/clipart_500_img_48_sent.mat');
    sentences = X.clipart_sentences;
    
    load('../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls');
    url = clipart_urls;

    parameters_dir = '../../data/search_parameters/clipart/cosine';

    n_images = length(clipart_urls);

    scores_w = []; scores_b = []; s = []; sent_pairs = [];
    for i=1:n_images
        progressbar(i-1, 10, n_images);
        split_url = strsplit(clipart_urls{i}, '/');
        filename = [split_url{end} '.mat'];
        
        X = load(sprintf('%s/mus/img_%s', parameters_dir, filename));
        scores_w = cat(1, scores_w, X.scores_w);

        X = load(sprintf('%s/mud/img_%s', parameters_dir, filename), 'scores_b', 'sent_pairs');
        scores_b = cat(1, scores_b, X.scores_b);
        sent_pairs = cat(3, sent_pairs, X.sent_pairs);
        
        X = load(sprintf('%s/s/ref0_query47/target_%s', parameters_dir, filename));
        s = cat(1, s, X.s);
    end
    
    m_sentences = 40; sent_pairs = permute(sent_pairs, [3 1 2]);
end

fprintf(' [Done]\n');

end