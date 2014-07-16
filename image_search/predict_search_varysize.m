function predict_search_varysize(features, method, predictor, dataset, n_jobs, runs)
% Author : Mainak Jas
%
% Predicts the specificity parameters and uses that to rank the
% images
%
% Parameters
% ----------
%
% features : str
%   'decaf' | 'objectness' | 'saliencymap' | 'decaf-objectness' | etc ...
% method : str
%   'rbf-svm' | 'linear-svm'
% predictor : str
%   'logistic' | 'gaussfit'
% dataset : str
%   'pascal' | 'clipart'
% n_jobs : int
%   number of jobs
% runs : int
%   number of runs

addpath(genpath('../../library/libsvm-3.17/'));
addpath(genpath('../../library/boundedline/'));
addpath('../aux_functions');

[scores_b, scores_w, s, sentences, ~, urls] = load_search_parameters(dataset);

% FIND SPECIFICITY
fprintf('\nCalculating specificity ... ');
if strcmpi(predictor, 'logistic')

    load('../../data/specificity_alldatasets.mat');
    eval(['y = specificity.' dataset '.B0;']);
    eval(['z = specificity.' dataset '.B1;']);

elseif strcmpi(predictor, 'gaussfit')
    eval(['y = specificity.' dataset '.mean;']);
end

fprintf(' [Done]');
s(isnan(s(:))) = -Inf;

% CLASSIFICATION FEATURES
fprintf('\nLoading classification image features ... ');
load(sprintf('../../data/image_features/feat_%s.mat',dataset));

X = [];
if regexpi(features, 'decaf')
    X = double(cat(2, Feat.decaf));
end

fprintf(' [Done]');

% PARAMETERS TO TRY
C = [0.01, 0.1, 1, 10, 100];

if strcmpi(method, 'rbf-svm')
    gamma = [0.0001, 0.0002, 0.0005, 0.001, 0.002];

    [param1, param2] = meshgrid(C, gamma);
    params = [param1(:) param2(:)]; clear param1 param2;
elseif strcmpi(method, 'linear-svm')
    params = C;
end

% TRAINING AND TEST SET SET
if strcmpi(dataset, 'pascal')
    train_stop = [10:10:800];
    test_idx = [801:1000];
elseif strcmpi(dataset, 'clipart')
    train_stop = [10:10:400];
    test_idx = [401:500];
end

% CALCULATE BASELINE
fprintf('\nSaving baseline ... ');
s_test = s(test_idx, test_idx);
for query_idx = 1:length(test_idx)
    [~, idx_b] = sort(s_test(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end
save(['../../data/predict_search/' dataset '/search_baseline.mat'], 'rank_b');
fprintf('[Done]');

% CALCULATING SPECIFICITY RANKINGS
fprintf('\nCalculating specificity rankings ... \n');
for run=1:runs
    for params_idx=1:length(params)
        filename = sprintf('../../data/predict_search/%s/search_params%d_run%d_%s_%s_%s.mat', ...
                           dataset, params_idx, run, features, method, predictor);
        if exist(filename, 'file')
            continue;
        end
        rank_s = zeros(length(train_stop), length(test_idx));
        for i=1:length(train_stop)

            if strcmpi(method, 'rbf-svm')
                fprintf('\nFEATURES = %s, METHOD = %s, PREDICTOR = %s \nRUN = %d, TRAINING SIZE = %d, C = %f, gamma = %f\n', ...
                        features, method, predictor, run, train_stop(i), params(params_idx, 1), params(params_idx, 2));
            elseif strcmpi(method, 'linear-svm')
                fprintf('\nFEATURES = %s, METHOD = %s, PREDICTOR = %s \nRUN = %d, TRAINING SIZE = %d, C = %f\n', ...
                        features, method, predictor, run, train_stop(i), params(params_idx));
            end
            
            % Select training data
            train_idx = randsample(train_stop(end), train_stop(i));

            % Normalize training features
            [Z_train, mu, sigma] = zscore(X(train_idx,:));

            % Train models for predicting specificity
            if strcmpi(method, 'rbf-svm')
                model_y = svmtrain(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', params(params_idx, 1), params(params_idx, 2)));
                if strcmpi(predictor, 'logistic')
                    model_z = svmtrain(z(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', params(params_idx, 1), params(params_idx, 2)));
                end
            elseif strcmpi(method, 'linear-svm')
                model_y = svmtrain(y(train_idx), Z_train, sprintf('-s 3 -t 0 -c %d -q', params(params_idx)));
                if strcmpi(predictor, 'logistic')
                    model_z = svmtrain(z(train_idx), Z_train, sprintf('-s 3 -t 0 -c %d -q', params(params_idx)));
                end
            end

            % Normalize the test features
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            % Predict specificity
            y_out = svmpredict(y(test_idx), Z_test, model_y, '-q');
            if strcmpi(predictor, 'logistic')
                z_out = svmpredict(z(test_idx), Z_test, model_z, '-q');
            end
            
            % MATCH QUERY SENTENCE WITH REFERENCE SENTENCES IN TEST SET
            mu_s = y_out;
            mu_d = 0.2; sigma_s = 0.1; sigma_d = 0.1;
            
            s_test = s(test_idx, test_idx);
            r_s = zeros(length(test_idx), length(test_idx)); r_d = r_s;
            ranks = zeros(1, length(test_idx));
            for query_idx=1:length(test_idx)
                for ref_idx=1:length(test_idx)

                    if strcmpi(predictor, 'gaussfit')
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