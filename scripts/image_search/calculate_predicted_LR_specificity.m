% CALCULATE_PREDICTED_LR_SPECIFICITY finds the predicted LR specificity
% using a Support Vector Regressor
%
% INPUT
%   data/image_features/feat_{dataset}.mat
%   data/predict_search/{dataset}/optimal_param_1000im.mat
%   data/search_parameters/{dataset}/LR/predicted_img_{filename}
%
% OUTPUT
%   data/search_parameters/{dataset}/predicted_LR.mat
%
% {dataset} = pascal / clipart
%
% AUTHOR: Mainak Jas
%
% See also: calculate_percentage_curve
function calculate_predicted_LR_specificity(dataset)

addpath(genpath('../../library/libsvm-3.17/'));
addpath('../io');

[~, ~, ~, sentences, ~, url] = load_search_parameters(dataset);
[n_images, ~] = size(sentences);

%% CLASSIFICATION FEATURES
fprintf('\nLoading classification image features ... ');
load(sprintf('../../data/image_features/feat_%s.mat',dataset));
fprintf('[Done]\n');

if strcmpi(dataset, 'clipart')
    X = double(cat(2, Feat.objectOccurence, Feat.type, Feat.objectCooccurence, ...
                   Feat.x, Feat.y, Feat.z, Feat.flip));
elseif strcmpi(dataset, 'pascal')
    X = double(cat(2, Feat.decaf));
end

%% LOADING PARAMETERS
fprintf('\nStarting grid search ...');
param_file = ['../../data/image_search/optimal_param_' dataset '.mat'];
load(param_file);

n_folds = n_images;
split = 1:n_images;

%% PREDICT SPECIFICITY
y_pred = zeros(1, n_images);
z_pred = zeros(1, n_images);
for fold_idx=1:n_folds

    split_url = strsplit(url{fold_idx}, '/');
    filename = split_url{end};

    load(sprintf('../../data/image_search/%s/LR_params/GT/predicted_img_%s.mat', dataset, filename));

    fprintf('SVR :: fold %d\n', fold_idx);

    y = B(:, 1); z = B(:, 2);
    train_idx = (split ~= fold_idx);
    test_idx = (split == fold_idx);

    [Z_train, mu, sigma] = zscore(X(train_idx,:));

    optimalc_y = bestc.y*sum(train_idx)/n_images;
    optimalc_z = bestc.z*sum(train_idx)/n_images;

    % Train models for predicting specificity
    model_y = svmtrain2(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', optimalc_y, bestg.y));
    model_z = svmtrain2(z(train_idx), Z_train, sprintf('-s 3 -c %d -g %f -q', optimalc_z, bestg.z));

    % Normalize the test features
    sigma0 = sigma;
    sigma0(sigma0==0) = 1;
    Z_test = bsxfun(@minus,X(test_idx,:), mu);
    Z_test = bsxfun(@rdivide, Z_test, sigma0);

    % Predict specificity
    y_pred(test_idx) = svmpredict2(y(test_idx), Z_test, model_y, '-q');
    z_pred(test_idx) = svmpredict2(z(test_idx), Z_test, model_z, '-q');
end

save(['../../data/image_search/' dataset '/LR_params/Pred/predicted_LR.mat'], ...
     'y_pred', 'z_pred');
end