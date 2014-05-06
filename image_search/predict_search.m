% Author: Mainak Jas

clear all;
addpath(genpath('../../library/libsvm-3.17/'));
experiment = 'singlereference'; % {'singlereference', 'multiplereferences'}

load('../../data/pascal_1000_img_50_sent.mat', 'pascal_urls');
load('../../data/image_search_50sentences_parameters.mat', 'scores_w');

if strcmpi(experiment, 'singlereference')
    load('../../data/image_search_50sentences_query.mat', 's');
else
    load('../../data/image_search_50sentences_query_refs.mat', 's');     
end

specificity = nanmean(scores_w, 2); s(isnan(s(:))) = -Inf;

% CLASSIFICATION FEATURES
for i=1:length(pascal_urls)
    filename = strsplit(pascal_urls{i}, '/');
    load(sprintf('../../data/pascal_decaf/%s_decaf.mat',cell2mat(filename(end))), 'fc6n');
    X(i, :) = double(fc6n); clear fc6n;
end

y = specificity;

folds = 5; C=1; gamma=0.001;
idx = crossvalind('Kfold',length(y), folds);

for i=1:1%folds
    
    fprintf('\nFOLD = %d\n\n', i)
    
    % SVM PREDICTION USING DECAF FEATURES
    %train_idx = find(idx~=i); test_idx = find(idx==i);
    train_idx = 1:1000; test_idx = 1:1000; % Uncomment later
    
    [Z_train,mu,sigma] = zscore(X(train_idx,:));
    
    model = svmtrain(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', C, gamma));
    
    sigma0 = sigma;
    sigma0(sigma0==0) = 1;
    Z_test = bsxfun(@minus,X(test_idx,:), mu);
    Z_test = bsxfun(@rdivide, Z_test, sigma0);
    
    y_out = svmpredict(y(test_idx), Z_test, model);
        
    % MATCH QUERY SENTENCE WITH REFERENCE SENTENCES IN TEST SET
    mu_s = y_out; mu_d = 0.2;
    sigma_s = 0.1; sigma_d = sigma_s;       
    
    % GAUSSIAN MIXTURE MODEL
    % obj = fitgmdist(scores_w);
    
    if strcmpi(experiment, 'singlereference')
        s_test = s(test_idx, test_idx);
    else
        s_test = s(1:4, test_idx, test_idx);
    end
    
    for query_idx=1:length(test_idx)
        fprintf('.');
        
        if strcmpi(experiment, 'singlereference')
            
            for ref_idx=1:length(test_idx)
                p_s = normpdf(s_test(query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                p_d = normpdf(s_test(query_idx, ref_idx), mu_d, sigma_d);
                
                r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                r_d(query_idx, ref_idx) = p_d/(p_s + p_d);                
               
            end
            
        else           

            for ref_idx=1:length(test_idx)

                p_s = 0; p_d = 0;

                for u=1:4
                    p_s = p_s + normpdf(s_test(u, query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                    p_d = p_d + normpdf(s_test(u, query_idx, ref_idx), mu_d, sigma_d);
                    
                end

                p_s = p_s/4; p_d = p_d/4; 
                
                r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                r_d(query_idx, ref_idx) = p_d/(p_s + p_d);                    

            end

        end
        
        r_s(isnan(r_s(:))) = -Inf;
        
        if strcmpi(experiment, 'singlereference')
            [~, idx_b] = sort(s_test(query_idx, :),'descend');
        else
            [~, idx_b] = sort(mean(s_test(1:4, query_idx, :),1), 'descend');
        end
        
        [~, idx_s] = sort(r_s(query_idx, :), 'descend');
        
        rank_b(i, query_idx) = find(idx_b==query_idx);
        rank_s(i, query_idx) = find(idx_s==query_idx);        
    end      
end

fprintf('\nMean specificity rank = %0.2f', mean(rank_s(:)));
fprintf('\nMean baseline rank = %0.2f', mean(rank_b(:)));