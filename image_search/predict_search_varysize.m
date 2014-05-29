% Author: Mainak Jas

clear all;
addpath(genpath('../../library/libsvm-3.17/'));
addpath(genpath('../../library/boundedline/'));

runs = 50;
features = 'decaf'; %{'decaf', 'objectness'}
method = 'svm-rbf'; %{'rbf-svm', 'linear-svm'}
predictor = 'logistic'; %{'logistic', 'naive'}

n_jobs = input('Please enter the number of jobs: ', 's');
dataset = input('Please enter the dataset (pascal/clipart): ', 's');

[scores_b, scores_w, s, sentences, ~, urls] = load_search_parameters(dataset);
[n_images, n_sentences] = size(sentences);

% FIND SPECIFICITY
for idx=1:n_images
    y_s = scores_w(idx,:);
    y_d = scores_b(idx,:);

    len = min(length(y_s), length(y_d));

    X = cat(2, y_s(1:len), y_d(1:len));
    labels = cat(1, ones(len,1), zeros(len,1));

    B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
end

clear X;

if strcmpi(predictor, 'logistic')
    y = B(:, 1);
    z = B(:, 2);
elseif strcmpi(predictor, 'naive')
    y = nanmean(scores_w, 2);
end

s(isnan(s(:))) = -Inf;

% CLASSIFICATION FEATURES
for i=1:n_images
    filename = strsplit(urls{i}, '/');
    mat = load(sprintf('../../data/image_features/%s/%s/%s_%s.mat', features, dataset, cell2mat(filename(end)), features));

    if strcmpi(features, 'decaf')
        X(i, :) = double(mat.fc6n);
    elseif strcmpi(features, 'objectness')
        X(i, :) = double(mat.obj_heatmap(:));
    end

end

C = [0.01, 0.1, 1, 10, 100];
gamma = [0.0001, 0.0002, 0.0005, 0.001, 0.002];

[param1, param2] = meshgrid(C, gamma);
params = [param1(:) param2(:)]; clear param1 param2;

if strcmpi(dataset, 'pascal')
    train_stop = [10:10:800];
    test_idx = [801:1000];
elseif strcmpi(dataset, 'clipart')
    train_stop = [10:10:400];
    test_idx = [401:500];
end

fprintf('\nSaving baseline ... ');
s_test = s(test_idx, test_idx);
for query_idx = 1:length(test_idx)
    [~, idx_b] = sort(s_test(query_idx, :),'descend');
    rank_b = find(idx_b==query_idx);
end
save(['../../data/predict_search/' dataset '/search_baseline.mat'], 'rank_b');
fprintf('[Done]');

matlabpool('open', n_jobs);

for run=1:runs
    for params_idx=1:length(params)
        filename = sprintf('../../data/predict_search/%s/search_params%d_run%d_%s.mat', dataset, params_idx, run, features);
        if exist(filename, 'file')
            continue;
        end
        rank_s = zeros(length(train_stop), length(test_idx));
        parfor i=1:length(train_stop)

            fprintf('\nRUN = %d, TRAINING SIZE = %d, C = %f, gamma = %f', ...
                    run, train_stop(i), params(params_idx, 1), params(params_idx, 2));

            % SVM PREDICTION USING DECAF FEATURES
            train_idx = randsample(train_stop(end), train_stop(i));
            
            [Z_train, mu, sigma] = zscore(X(train_idx,:));

            % Train models for both parameters of the logistic regression model
            model_y = svmtrain(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', params(params_idx, 1), params(params_idx, 2)));
            model_z = svmtrain(z(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', params(params_idx, 1), params(params_idx, 2)));
            
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            y_out = svmpredict(y(test_idx), Z_test, model_y, '-q');
            z_out = svmpredict(z(test_idx), Z_test, model_z, '-q');
            
            % MATCH QUERY SENTENCE WITH REFERENCE SENTENCES IN TEST SET
            mu_s = y_out; mu_d = 0.2;
            sigma_s = 0.1; sigma_d = sigma_s;
            
            s_test = s(test_idx, test_idx);
            r_s = zeros(length(test_idx), length(test_idx)); r_d = r_s;
            ranks = zeros(1, length(test_idx));
            for query_idx=1:length(test_idx)
                for ref_idx=1:length(test_idx)

                    if strcmpi(predictor, 'naive')
                        p_s = normpdf(s_test(query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                        p_d = normpdf(s_test(query_idx, ref_idx), mu_d, sigma_d);

                        r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                        r_d(query_idx, ref_idx) = p_d/(p_s + p_d);

                    elseif strcmpi(predictor, 'logistic')
                        r_s(query_idx, ref_idx) = glmval([y_out(ref_idx), z_out(ref_idx)]', s(query_idx, ref_idx), 'logit');
                    end

                end

                r_s(isnan(r_s(:))) = -Inf;

                [~, idx_s] = sort(r_s(query_idx, :), 'descend');
                ranks(query_idx) = find(idx_s==query_idx);

            end
            rank_s(i, :) = ranks;
        end
    fprintf('\n\tSaving %s ... ', filename);
    save(filename, 'rank_s');
    fprintf('[Done]');
    end
end

matlabpool('close');