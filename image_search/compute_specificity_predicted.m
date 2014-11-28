function compute_specificity_predicted(dataset, crossval)

predictor = 'logistic';
addpath(genpath('../../library/libsvm-3.17/'));

load('../../data/specificity_alldatasets.mat');
eval(['y = specificity.' dataset '.B0;']);
eval(['z = specificity.' dataset '.B1;']);
n_images = length(y);

% CLASSIFICATION FEATURES
fprintf('\nLoading classification image features ... ');
load(sprintf('../../data/image_features/feat_%s.mat',dataset));
fprintf('[Done]\n');

% GRID SEARCH
fprintf('\nStarting grid search ...');
X = double(cat(2, Feat.decaf));
param_file = ['../../data/predict_search/' dataset '/optimal_param_1000im.mat'];

if ~exist(param_file, 'file')
    Z = zscore(X);
    [bestc.y, bestg.y] = grid_search(Z, y);
    [bestc.z, bestg.z] = grid_search(Z, z);
    save(param_file, 'bestc', 'bestg', 'predictor');
else
    load(param_file);
end
fprintf(' [Done]\n\n');

if strcmpi(crossval, '5fold')
    % GENERATE RANDOM 5-fold SPLITS
    split_file = ['../../data/predict_search/' dataset '/prediction_splits.mat'];
    n_splits=25; n_folds = 5;
    if ~exist(split_file, 'file')
        split = zeros(n_splits, n_images);
        for split_idx=1:n_splits
            split(split_idx, :) = crossvalind('Kfold', n_images, n_folds);
        end
        save(split_file, 'split');
    else
        load(split_file);
    end
else
    n_splits = 1; n_folds = n_images;
    split = 1:n_images;
end

% PREDICT SPECIFICITY
y_pred = zeros(n_splits, n_images);
z_pred = zeros(n_splits, n_images);

for run=1:n_splits
    y_out = zeros(1, n_images);
    z_out = zeros(1, n_images);
    for fold_idx=1:n_folds
        fprintf('SVR :: run %d fold %d\n', run, fold_idx);
        train_idx = (split(run, :) ~= fold_idx);
        test_idx = (split(run, :) == fold_idx);        

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
        y_out(test_idx) = svmpredict2(y(test_idx), Z_test, model_y, '-q');
        z_out(test_idx) = svmpredict2(z(test_idx), Z_test, model_z, '-q');        
    end
    y_pred(run, :) = y_out;
    z_pred(run, :) = z_out;
end
save(['../../data/predict_search/' dataset '/predicted_specificity_' crossval '.mat'], ...
     'y_pred', 'z_pred');

end