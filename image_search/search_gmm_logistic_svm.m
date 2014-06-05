% Image search
clear all; warning off; close all;
addpath(genpath('../../library/boundedline/'));

method = 'svm'; % {'gmm', 'logistic', 'svm', 'linear-svm'}
dataset = 'pascal'; % {'pascal', 'memorability', 'clipart'}
expt = 'gridsearch'; % {'multiruns', 'singlerun', 'gridsearch'}

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

if strcmpi(expt, 'gridsearch')

    if strcmpi(method, 'svm')

        C = [0.01, 0.1, 1, 10, 100];
        gamma = [0.1, 0.5, 1, 2, 3, 5, 10];

        [param1, param2] = meshgrid(C, gamma);
        params = [param1(:) param2(:)]; clear param1 param2;
        outerloop = 1:length(params);
    elseif strcmpi(method, 'linear-svm') && strcmpi(expt, 'gridsearch')

        C = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
        outerloop = 1:length(C);

    end

elseif strcmpi(expt, 'multiruns')
    outerloop = [1:50];
end

cd('/home/mainak/Desktop/projects/specificity/library/libsvm-3.17/matlab/');

[n_images, n_sentences] = size(sentences); 

% mycluster = parcluster('local'); delete(mycluster.Jobs);
% poolobj = parpool;

for outerloop_idx=outerloop
        
    for n_tr=50%n_sentences%2:n_sentences
        
        fprintf('\n%s = %d, TRAINING SENTENCES = %d\n', expt, outerloop_idx, n_tr);
        
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

                model{outerloop_idx, idx} = svmtrain(labels, X', sprintf('-b 1 -c %f -g %f -q', params(outerloop_idx, 1), params(outerloop_idx, 2)));

            elseif strcmpi(method, 'linear-svm')

               len = min(length(y_s), length(y_d));

               X = cat(2, y_s(1:len), y_d(1:len));
               labels = cat(1, ones(len,1), zeros(len,1));

               model{outerloop_idx, idx} = svmtrain(labels, X', sprintf('-b 1 -t 0 -c %f -q', C(outerloop_idx)));

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
                    
                elseif strcmpi(method, 'svm') || strcmpi(method, 'linear-svm')
                    y_test = 1; % set this randomly, does not matter
                    [~, ~, prob_estimates] = svmpredict(y_test, s(query_idx, ref_idx), model{outerloop_idx, ref_idx}, '-b 1 -q');
                    r_s(query_idx, ref_idx) = prob_estimates(1);
                
                end
                                
            end
            
            r_s(isnan(r_s(:))) = -Inf;           
           
            % RANKING:: SPECIFICITY
            [~, idx] = sort(r_s(query_idx, :), 'descend');
            rank_s(outerloop_idx, n_tr, query_idx) = find(idx==query_idx);
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

% Show the results of gridsearch
if strcmpi(method, 'svm')
    reshape(mean(rank_s(1:30, 50, :), 3),6,5)
elseif strcmpi(method, 'linear-svm')
    mean(rank_s(1:7, 50, :), 3)
end

cd('/home/mainak/Desktop/projects/specificity/scripts/image_search/');