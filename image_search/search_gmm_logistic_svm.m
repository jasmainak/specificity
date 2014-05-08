% Image search
clear all; warning off; close all;
addpath(genpath('../../library/boundedline/'));

method = {'logistic'}; % {'gmm', 'logistic', 'svm'}
dataset = {'pascal'}; % {'pascal', 'memorability', 'clipart'}

if strcmpi(dataset, 'pascal')
    load('../../data/image_search_50sentences_query.mat');
    load('../../data/image_search_50sentences_parameters.mat');
    
    scores_b = [];
    for i=1:1000
        X = load(sprintf('../../data/search_parameters/pascal/mu_d/image_search_50sentences_mud_%d.mat', i - 1), 'scores_b');
        scores_b = cat(1, scores_b, X.scores_b);
    end
    clear X;
    
    m_sentences = 24;
end

cd('/home/mainak/Desktop/projects/specificity/library/libsvm-3.17/matlab/');

[n_images, n_sentences] = size(sentences); 

% mycluster = parcluster('local'); delete(mycluster.Jobs);
% poolobj = parpool;

for run=1:1%4
        
    for n_tr=50%n_sentences%2:n_sentences
        
        fprintf('\nRUN = %d, TRAINING SENTENCES = %d\n', run, n_tr);
        
        if n_tr<=48
            choose_sent = randsample(2:n_sentences, n_tr);
        elseif n_tr==49 % include ref sentence
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
        
        % TRAINING PHASE
        mu_s = zeros(1,length(sentences));
        
        fprintf('\nTRAINING PHASE ');
        
        for idx=1:length(sentences)
            
            if rem(idx,10)==0
                fprintf('.');
            end
            
            y_s = scores_w(idx,train_idx);
            y_d = scores_b(idx, 1:m_sentences*(n_tr-1));
                        
            if strcmpi(method, 'gmm')                
                gmm_s{idx} = fitgmdist(y_s', 10, 'SharedCov', true);
                gmm_d{idx} = fitgmdist(y_d', 10, 'SharedCov', true);
                
            elseif strcmpi(method, 'logistic')                      
                X = cat(2, y_s(1:length(y_d)), y_d);
                labels = cat(1, ones(length(y_d),1), zeros(length(y_d),1));
                B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
                
            elseif strcmpi(method, 'svm')                                              
                
                len = min(length(y_s), length(y_d));
                
                X = cat(2, y_s(1:len), y_d(1:len));
                labels = cat(1, ones(len,1), zeros(len,1));
                
                model{idx} = svmtrain(labels, X', sprintf('-b 1 -q'));
            end
                        
        end
                        
        % TEST PHASE
        r_s = zeros(n_images, n_images); r_d = r_s;
        
        fprintf('\nTEST PHASE ');
        for query_idx = 1:n_images                       
            
            if rem(query_idx,10)==0
                fprintf('.');
            end
            
            for ref_idx=1:n_images
                
                if strcmpi(method, 'gmm')
                    p_s = pdf(gmm_s{idx}, s(query_idx, ref_idx));
                    p_d = pdf(gmm_d{idx}, s(query_idx, ref_idx));
                    
                    r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                    
                elseif strcmpi(method, 'logistic')
                    r_s(query_idx, ref_idx) = glmval(squeeze(B(ref_idx, :))', s(query_idx, ref_idx), 'logit');
                    
                elseif strcmpi(method, 'svm')                    
                    y_test = 1; % set this randomly, does not matter
                    [~, ~, prob_estimates] = svmpredict(y_test, s(query_idx, ref_idx), model{ref_idx}, '-b 1');
                    r_s(query_idx, ref_idx) = prob_estimates(1);
                
                end
                                
            end
            
            r_s(isnan(r_s(:))) = -Inf;           
           
            % RANKING:: SPECIFICITY
            [~, idx] = sort(r_s(query_idx, :), 'descend');          
            rank_s(run, n_tr, query_idx) = find(idx==query_idx);            
        end
         
    end
    
end

% delete(poolobj);

% RANKING:: BASELINE
s(isnan(s(:))) = -Inf; 
for query_idx = 1:n_images    
    [~, idx_b] = sort(s(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end

mean(rank_s(1,48,:))

cd('/home/mainak/Desktop/projects/specificity/scripts/image_search/');