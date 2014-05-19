% Image search
clear all; close all;

addpath(genpath('../../library/boundedline/'));

% GET USER INPUTS
dataset = input('Enter the dataset (pascal / clipart / memorability): ', 's');
n_jobs = input('Enter number of jobs: ');

method = 'logistic';

% LOAD DATASET SEARCH PARAMETERS
if strcmpi(dataset, 'pascal')
    load('../../data/image_search_50sentences_query.mat');
    load('../../data/image_search_50sentences_parameters.mat');

    scores_b = [];
    for i=1:1000
        X = load(sprintf('../../data/search_parameters/pascal/mu_d/image_search_50sentences_mud_%d.mat', i - 1), 'scores_b');
        scores_b = cat(1, scores_b, X.scores_b);
    end

    m_sentences = 24;

elseif strcmpi(dataset, 'clipart')

   X = load('../../data/sentences/clipart_500_img_48_sent.mat');
   sentences = X.clipart_sentences;

   load('../../data/search_parameters/clipart/image_search_mud.mat');

   scores_w = [];
   for i=0:curr_idx
       X = load(sprintf('../../data/search_parameters/clipart/mus/image_search_mus_%d.mat',i));
       scores_w = cat(1, scores_w, X.scores_w);
   end

   scores_b = [];
   for i=0:curr_idx
       X = load(sprintf('../../data/search_parameters/clipart/mu_d/image_search_50sentences_mud_%d.mat',i));
       scores_b = cat(1, scores_b, X.scores_b);
   end

   s = [];
   for i=0:curr_idx
       X = load(sprintf('../../data/search_parameters/clipart/s/image_search_s_%d.mat', i));
       s = cat(1, s, X.s);
   end

   m_sentences = 23;
end

clear X curr_idx i;

[n_images, n_sentences] = size(sentences);
comb = combntns(1:n_sentences, 2);

matlabpool('open', n_jobs);
pctRunOnAll warning off;

for run=1:50
        
    for n_tr=2:n_sentences
        
        filename = sprintf('../../data/search_results/%s/ranks_%s_run%d_ntr%d.mat', dataset, method, run, n_tr);

        if exist(filename, 'file')
            continue;
        end

        if n_tr<= (n_sentences - 2)
            choose_sent = randsample(2:n_sentences, n_tr);
        elseif n_tr== (n_sentences - 1) % include ref sentence
            choose_sent = randsample(1:n_sentences, n_tr);
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
        parfor idx=1:n_images
            y_s = scores_w(idx,train_idx);
            y_d = scores_b(idx,1:m_sentences*(n_tr-1));

            len = min(length(y_s), length(y_d));

            X = cat(2, y_s(1:len), y_d(1:len));
            labels = cat(1, ones(len,1), zeros(len,1));

            B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
        end

        fprintf('Test phase ...');
        % TEST PHASE
        r_s = zeros(n_images, n_images); r_d = r_s;
        rank_s = zeros(1,n_images);
        
        for query_idx = 1:n_images

            if rem(query_idx, 10)==0
                fprintf('.');
            end
            
            parfor ref_idx=1:n_images
                r_s(query_idx, ref_idx) = glmval(squeeze(B(ref_idx, :))', s(query_idx, ref_idx), 'logit');
            end
            
            r_s(isnan(r_s(:))) = -Inf;           
           
            % RANKING:: SPECIFICITY        
            [~, idx_s] = sort(r_s(query_idx, :),'descend');                         
            rank_s(query_idx) = find(idx_s==query_idx);
        end

        fprintf('\n Saving %s ... ', filename);
        save(filename, 'rank_s');
        fprintf('[Done]\n');

    end
    
end

matlabpool('close');

% RANKING:: BASELINE
s(isnan(s(:))) = -Inf; 
for query_idx = 1:n_images    
    [~, idx_b] = sort(s(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end

% rank_s = rank_s(1:12, :, :);

rank_s_mean = squeeze(mean(rank_s,1));
rank_s_err = std(mean(rank_s,3),0,1);

% figure;
% x_range = 2:n_sentences;
% boundedline(x_range, mean(rank_s_mean(x_range,:),2), rank_s_err(x_range));
% h1 = plot(x_range, mean(rank_s_mean(x_range,:),2)); hold on;
% h2 = plot([2,n_sentences], [mean(rank_b), mean(rank_b)],'r--');
% plot(x_range, mean(rank_s_mean(x_range, :),2), 'bo', 'MarkerFaceColor', 'w');
% plot(49, mean(rank_s_mean(49, :),2), 'go', 'MarkerFaceColor','g');
% plot(50, mean(rank_s_mean(50, :),2), 'ro', 'MarkerFaceColor','r');
%
% set(gca, 'TickDir','out','Box','off','XTick',[10:10:50], ...
%     'XTickLabel',{'10C2','20C2','30C2','40C2','50C2'}, 'Fontsize',12);
%
% ylabel('Mean Rank','Fontsize',12); xlabel('Number of training sentences','Fontsize',12);
% legend([h1, h2], 'Specificity','Baseline','Fontsize',12);
% title('Effect of changing number of training sentences','Fontsize',14);