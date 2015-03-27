% Image Search
%
% Author: Mainak Jas

clear all; close all;
addpath('../io/');

%% GET USER INPUTS
dataset = input('Enter the dataset (pascal / clipart): ', 's');
n_jobs = input('Enter number of jobs: ');

[scores_b, scores_w, s, sentences, m_sentences] = load_search_parameters(dataset);

[n_images, n_sentences] = size(sentences);
comb = combntns(1:n_sentences, 2);

%% RANKING:: BASELINE
s(isnan(s(:))) = -Inf;
for query_idx = 1:n_images
    [~, idx_b] = sort(s(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end

fprintf('Saving baseline ... ');
save(['../../data/search_results/' dataset '/ranks_baseline.mat'],'rank_b');
fprintf('[Done]');

matlabpool('open', n_jobs);
pctRunOnAll warning off;

%% RANKING:: SPECIFICITY
for run=1:25  
    for n_tr=2:n_sentences
        
        filename = sprintf('../../data/search_results/%s/ranks_logistic_run%d_ntr%d.mat', dataset, run, n_tr);

        if exist(filename, 'file')
            continue;
        end

        if n_tr<= (n_sentences - 2)
            choose_sent = randsample(2:n_sentences-1, n_tr);
        elseif n_tr== (n_sentences - 1) % include ref sentence
            choose_sent = randsample(1:n_sentences-1, n_tr);
        else % include ref + query sentence
            choose_sent = randsample(1:n_sentences, n_tr);
        end

        mask = zeros(length(comb), 1);
        pairs = nchoosek(choose_sent, 2);
         
        for i=1:size(pairs,1)
            mask = (comb(:,1)==pairs(i,1) & comb(:,2)==pairs(i,2)) | (comb(:,2)==pairs(i,1) & comb(:,1)==pairs(i,2)) | mask;
        end

        train_idx = find(mask == 1);               
        
        B = zeros(n_images, 2);
        % TRAINING PHASE
        fprintf('Training phase ...\n');
        for idx=1:n_images
            y_s = scores_w(idx,train_idx);
            y_d = scores_b(idx,1:m_sentences*(n_tr-1));

            len = min(length(y_s), length(y_d));

            X = cat(2, y_s(1:len), y_d(1:len));
            labels = cat(1, ones(len,1), zeros(len,1));

            B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
        end

        fprintf('Test phase ...');
        % TEST PHASE
        rank_s = zeros(1,n_images);
        parfor query_idx = 1:n_images

            r_s = zeros(n_images, 1);

            for ref_idx=1:n_images
                r_s(ref_idx) = glmval(squeeze(B(ref_idx, :))', s(query_idx, ref_idx), 'logit');
            end
            
            r_s(isnan(r_s(:))) = -Inf;           
           
            % RANKING:: SPECIFICITY        
            [~, idx_s] = sort(r_s,'descend');                         
            rank_s(query_idx) = find(idx_s==query_idx);
        end

        fprintf('\n Saving %s ... ', filename);
        save(filename, 'rank_s');
        fprintf('[Done]\n');

    end
end

matlabpool('close');