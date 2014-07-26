function predict_search_vary_numsentences(features, predictor, dataset, runs, overwrite)
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
% predictor : str
%   'logistic' | 'gaussfit' | 'groundtruth'
% dataset : str
%   'pascal' | 'clipart'
% runs : int
%   number of runs
% overwrite : bool
%   overwrite existing files?

addpath(genpath('../../library/libsvm-3.17/'));
addpath(genpath('../../library/boundedline/'));
addpath('../aux_functions');

[~, ~, s, ~, ~, ~] = load_search_parameters(dataset);

% FIND SPECIFICITY
fprintf('\nCalculating specificity ... ');
if strcmpi(predictor, 'logistic') || strcmpi(predictor, 'groundtruth')

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

% TRAINING AND TEST SET SET
if strcmpi(dataset, 'pascal')
    train_size = [10:10:800];
    test_idx = [801:1000];
elseif strcmpi(dataset, 'clipart')
    train_size = [10:10:400];
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

% GRID SEARCH

if ~strcmpi(predictor, 'groundtruth')
    param_file = ['../../data/predict_search/' dataset '/optimal_param.mat'];

    Z_train = zscore(X(1:train_size(end), :));

    if ~exist(param_file, 'file')
        [bestc.y, bestg.y] = grid_search(Z_train, y, train_size);
        if strcmpi(predictor, 'logistic')
            [bestc.z, bestg.z] = grid_search(Z_train, z, train_size);
        end
        save(param_file, 'bestc', 'bestg', 'predictor');
    else
        load(param_file);
    end
else
    bestc.y = NaN; bestc.z = NaN; bestg.y = NaN; bestg.z = NaN;
end

% CALCULATING SPECIFICITY RANKINGS
fprintf('\nCalculating specificity rankings ... \n');
for run=1:runs

    filename = sprintf('../../data/predict_search/%s/search_run%d_%s_%s.mat', ...
        dataset, run, features, predictor);
    if exist(filename, 'file') && ~overwrite
       continue;
    end

    rank_s = zeros(length(train_size), length(test_idx));
    for i=1:length(train_size)

        fprintf('\nFEATURES = %s, PREDICTOR = %s \nRUN = %d, TRAINING SIZE = %d, C = %f, gamma = %f\n', ...
                features, predictor, run, train_size(i), bestc.y, bestg.y);

        % Select training data
        train_idx = randsample(train_size(end), train_size(i));

        if ~strcmpi(predictor, 'groundtruth')
            % Normalize training features
            [Z_train, mu, sigma] = zscore(X(train_idx,:));

            optimalc_y = bestc.y*train_size(i)/train_size(end);
            optimalc_z = bestc.z*train_size(i)/train_size(end);
            % Train models for predicting specificity
            model_y = svmtrain2(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', optimalc_y, bestg.y));
            if strcmpi(predictor, 'logistic')
                model_z = svmtrain2(z(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', optimalc_z, bestg.z));
            end

            % Normalize the test features
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);

            % Predict specificity
            y_out = svmpredict2(y(test_idx), Z_test, model_y, '-q');
            if strcmpi(predictor, 'logistic')
                z_out = svmpredict2(z(test_idx), Z_test, model_z, '-q');
            end
        else
            z_out = z(test_idx);
            y_out = y(test_idx);
        end

        % MATCH QUERY SENTENCE WITH REFERENCE SENTENCES IN TEST SET
        mu_s = y_out;
        mu_d = 0.2; sigma_s = 0.1; sigma_d = 0.1;

        s_test = s(test_idx, test_idx);
        r_s = zeros(length(test_idx), length(test_idx)); r_d = r_s;
        ranks = zeros(1, length(test_idx));
        for query_idx=1:length(test_idx)
            progressbar(query_idx, 10, length(test_idx));
            for ref_idx=1:length(test_idx)

                if strcmpi(predictor, 'gaussfit')
                    p_s = normpdf(s_test(query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                    p_d = normpdf(s_test(query_idx, ref_idx), mu_d, sigma_d);

                    r_s(query_idx, ref_idx) = p_s/(p_s + p_d);
                    r_d(query_idx, ref_idx) = p_d/(p_s + p_d);

                elseif strcmpi(predictor, 'logistic') || strcmpi(predictor, 'groundtruth')
                    r_s(query_idx, ref_idx) = glmval([y_out(ref_idx), z_out(ref_idx)]', s_test(query_idx, ref_idx), 'logit');
                end

            end

            r_s(isnan(r_s(:))) = -Inf;

            [~, idx_s] = sort(r_s(query_idx, :), 'descend');
            ranks(query_idx) = find(idx_s==query_idx);

        end
        rank_s(i, :) = ranks;  % ranks for one training size
    end
    fprintf('\n\tSaving %s ... ', filename);
    save(filename, 'rank_s');
    fprintf('[Done]');
end

end

function [bestc, bestg] = grid_search(Z_train, y, train_size)

% Grid search to select C and gamma

optimalg = 1/size(Z_train,2); % 1/number of features
bestcv = Inf;
for log10C=-1:3
    for g = optimalg/2:optimalg/10:optimalg*1.5
        cv = svmtrain2(y(1:train_size(end)), Z_train, ['-s 3 -v 5 -q -c ' num2str(10^log10C) ' -g ', num2str(g)]);
        if (cv < bestcv),
            bestcv = cv; bestc = 10^log10C; bestg = g;
        end
        fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)', 10^log10C, g, cv, bestc, bestg, bestcv);
    end
end

% Refine grid search
for C=bestc/2:bestc/10:bestc*1.5
    for g = optimalg/2:optimalg/10:optimalg*1.5
        cv = svmtrain2(y(1:train_size(end)), Z_train, ['-s 3 -v 5 -q -c ' num2str(C) ' -g ', num2str(g)]);
        if (cv < bestcv),
            bestcv = cv; bestc = C; bestg = g;
        end
        fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)', C, g, cv, bestc, bestg, bestcv);
    end
end

end