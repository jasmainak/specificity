% Create sentence pairs for similarity analysis

clear all; close all;

proj_dir = '/home/mainak/Desktop/specificity';
addpath([proj_dir '/library/cvpr_memorability_code/Code/Library/cell2csv']);

load('../data/sentence_descriptions.mat');

[n_images, n_sentences] = size(sentences);
n_pairs = nchoosek(n_sentences,2)*n_images;

pairs = cell(n_pairs, 2); 
mappings = zeros(n_pairs, 3);

% create sentence pairs
u = 1; 
for i=1:n_images
    for j=1:n_sentences-1
        for k=j+1:n_sentences
            pairs{u,1} = ['"' sentences{i, j} '"'];
            pairs{u,2} = ['"' sentences{i, k} '"'];
            mappings(u, :) = [i, j, k];
            u = u + 1;
        end
    end
end

% randomize sentence pairs
rand('state', 42);
randomorder = randperm(n_pairs);

pairs = pairs(randomorder, :);

% write out csv file for MTurk
pair_headers = {'p1_sent1', 'p1_sent2', 'im1', 'im1_s1', 'im1_s2',...
                'p2_sent1', 'p2_sent2', 'im2', 'im2_s1', 'im2_s2',...
                'p3_sent1', 'p3_sent2', 'im3', 'im3_s1', 'im3_s2'};

% concatenate meta-information to sentences
pairs = cat(2, pairs, num2cell(mappings(randomorder, :)));

% reshape data to contain 3 pairs per HIT
pairs = reshape(pairs', size(pairs,2)*n_pairs, 1);
pairs = cat(1, pair_headers, reshape(pairs, length(pair_headers), ...
            length(pairs)/length(pair_headers))');

cell2csv('../data/mturk/input/mturk_similarity.csv', pairs, ',');