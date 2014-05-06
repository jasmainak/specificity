% Author: Mainak Jas

clear all;
addpath(genpath('../../library/libsvm-3.17/'));
addpath(genpath('../../library/boundedline/'));

experiment = 'singlereference'; % {'singlereference', 'multiplereferences'}

load('../../data/pascal_1000_img_50_sent.mat', 'pascal_urls');
load('../../data/image_search_50sentences_parameters.mat', 'scores_w');
load('../../data/image_search_50sentences_query.mat', 's');

specificity = nanmean(scores_w, 2); s(isnan(s(:))) = -Inf;

% CLASSIFICATION FEATURES
for i=1:length(pascal_urls)
    filename = strsplit(pascal_urls{i}, '/');
    load(sprintf('../../data/pascal_decaf/%s_decaf.mat',cell2mat(filename(end))), 'fc6n');
    X(i, :) = double(fc6n); clear fc6n;
end

y = specificity;

C=1; gamma=0.001;
train_stop = [10:10:800];
test_idx = [801:1000];

poolobj = parpool;
rank_b = zeros(20, length(train_stop), length(test_idx));
rank_s = rank_b;

parfor run=1:50
    rank_b_temp = zeros(length(train_stop), length(test_idx));
    rank_s_temp = rank_b_temp;
    for i=1:length(train_stop)
        
        fprintf('\nRUN = %d, TRAINING SIZE = %d\n\n', run, train_stop(i));
        
        % SVM PREDICTION USING DECAF FEATURES
        %train_idx = 1:train_stop(i);
        train_idx = randsample(1:800, train_stop(i));
        
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
        
        s_test = s(test_idx, test_idx);
        r_s = zeros(length(test_idx), length(test_idx)); r_d = r_s;
        for query_idx=1:length(test_idx)
            
            for ref_idx=1:length(test_idx)
                p_s = normpdf(s_test(query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                p_d = normpdf(s_test(query_idx, ref_idx), mu_d, sigma_d);
                
                r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                r_d(query_idx, ref_idx) = p_d/(p_s + p_d);
            end
            
            r_s(isnan(r_s(:))) = -Inf;
            
            [~, idx_b] = sort(s_test(query_idx, :),'descend');
            [~, idx_s] = sort(r_s(query_idx, :), 'descend');
            
            rank_b_temp(i, query_idx) = find(idx_b==query_idx);
            rank_s_temp(i, query_idx) = find(idx_s==query_idx);
            
        end
    end
    rank_b(run, :, :) = rank_b_temp;
    rank_s(run, :, :) = rank_s_temp;
end

delete(poolobj);
meanrank_s = squeeze(mean(rank_s, 3));

boundedline(train_stop, mean(meanrank_s), std(meanrank_s)); hold on;
h1 = plot(train_stop, mean(meanrank_s)); hold on; 
plot(train_stop, mean(meanrank_s), 'bo', 'MarkerFacecolor','w','Markersize',10);

h2 = plot([10, train_stop(end)], [mean(rank_b(:)) mean(rank_b(:))], 'r--');
ylabel('Mean rank','Fontsize',12); xlabel('Training data size','Fontsize',12);
legend([h1, h2], 'Specificity', 'Baseline');
set(gca,'Tickdir','out','Box','off','Fontsize',12);