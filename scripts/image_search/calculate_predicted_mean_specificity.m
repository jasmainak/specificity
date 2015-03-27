% Calculate predicted mean specificity using SVR
% Author: Mainak Jas
function calculate_predicted_mean_specificity()

addpath(genpath('../../library/libsvm-3.17/'));
rng('default');  % to avoid surprises

features = 'decaf';
try_features(features, 'memorability', 'mean', 'vary_size');
try_features(features, 'pascal', 'mean', 'vary_size');

features = 'objectOccurence-objectcoOccurence-xyz-flip-type';
try_features(features, 'clipart', 'mean', 'vary_size');

end

function try_features(features, dataset, specificity_type, experiment)

% Load features

fprintf('\nLoading %s dataset ... ', dataset);
load(sprintf('../../data/image_features/feat_%s.mat',dataset));
fprintf(' [Done]');

% Labels for prediction (specificity)

load('../../data/specificity_alldatasets.mat');
eval(['y = specificity.' dataset '.' specificity_type ';']);

% Feature Vector

X = [];

if regexpi(features, 'gist')
    if strcmpi(dataset, 'memorability')
        load('../../data/memorability_mapping.mat');
        X = cat(2, X, double(Feat.gist(mapping, :)));
    elseif strcmpi(dataset, 'pascal')
        X = cat(2, X, double(Feat.gist));
    end
end

if regexpi(features, 'decaf')
    X = double(cat(2, X, Feat.decaf));
end

if regexpi(features, 'objectOccurence')
    X = double(cat(2, X, Feat.objectOccurence));
end

if regexpi(features, 'objectcoOccurence')
    X = double(cat(2, X, Feat.objectCooccurence));
end

if regexpi(features, 'type')
    X = double(cat(2, X, Feat.type));
end

if regexpi(features, 'pose')
    X = double(cat(2, X, Feat.mike_pose));
    X = double(cat(2, X, Feat.jenny_pose));
end

if regexpi(features, 'expression')
    X = double(cat(2, X, Feat.mike_expression));
    X = double(cat(2, X, Feat.jenny_expression));
end

if regexpi(features, 'x')
    X = double(cat(2, X, Feat.x));
end

if regexpi(features, 'y')
    X = double(cat(2, X, Feat.y));
end

if regexpi(features, 'z')
    X = double(cat(2, X, Feat.z));
end

if regexpi(features, 'flip')
    X = double(cat(2, X, Feat.flip));
end

if strcmpi(dataset, 'pascal') || strcmpi(dataset, 'clipart')
    y = y(~isnan(y));
    X = X(~isnan(y), :);
end

% Vary size of training data

if strcmpi(dataset, 'memorability')
    train_size = 100:100:700; test_idx = 701:888;
elseif strcmpi(dataset, 'pascal')
    train_size = 100:100:800; test_idx = 801:length(y);
elseif strcmpi(dataset, 'clipart')
    train_size = 50:50:400; test_idx = 401:499;
end

r_spearman = zeros(length(train_size), 50);

if regexpi(experiment, 'grid_search')
    
    % Grid search to select C and gamma
    Z_train = zscore(X(1:train_size(end), :));
    [bestc, bestg] = grid_search(Z_train, y(1:train_size(end)));
else
    bestc = 1;
    bestg = 1/size(X, 2);
end

if regexpi(experiment, 'vary_size')

    for run=1:50
        
        randomorder = randperm(length(y));
        X = X(randomorder, :);
        y = y(randomorder);
        
        for i=1:length(train_size)
            
            fprintf('\nFeature = %s, Run %d, Trainsize = %d\n', features, run, train_size(i));

            train_idx = 1:train_size(i); 
            % test_idx = train_idx; % uncomment later
            
            optimalc = bestc*train_size(i)/train_size(end);
            
            [Z_train,mu,sigma] = zscore(X(train_idx,:));
            
            model = svmtrain2(y(train_idx), Z_train, ['-s 3 -q -c ' num2str(optimalc) ' g ' num2str(bestg)]);
            
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            y_out = svmpredict2(y(test_idx), Z_test, model, '-q');     
            r_spearman(i, run) = corr(y_out, y(test_idx), 'type', 'spearman');
            
        end
        
    end
    
end

save(sprintf('../../data/predict_specificity/%s_%s.mat', dataset, features), ...
     'r_spearman', 'train_size');

end