% Clear variables and add paths

function predict_specificity_tryfeatures()

addpath(genpath('../../library/libsvm-3.17/'));

rng('default');  % to avoid surprises

%features = {'decaf'};
%n_features = length(features);

%for i=1:n_features
%    try_features(features{i}, 'memorability', 'mean', 'vary_size');
%end

%for i=1:n_features
%   try_features(features{i}, 'pascal', 'mean', 'vary_size');
%end

features = {'objectOccurence-objectcoOccurence-xyz-flip-type-pose-expression'};
n_features = length(features);

for i=1:n_features
    try_features(features{i}, 'clipart', 'mean', 'vary_size');
end

% 
% figure;
% features = {'gist', 'attributes', 'decaf', 'saliencymap', 'objectness'};
% n_features = length(features);
% colors = hsv(n_features);
% 
% subplot(2,2,1);
% for i=1:n_features
%     try_features(features{i}, 'memorability', 'mean', 'vary_size_grid_search', colors(i, :));
%     legend(features(1:i), 'Location', 'BestOutside'); drawnow;
% end
% title('Specificity Prediction using SVR (Memorability dataset)[mean]');
% 
% features = {'gist', 'decaf', 'saliencymap', 'objectness'};
% n_features = length(features);
% colors = hsv(n_features);
% 
% subplot(2,2,2);
% for i=1:n_features
%     try_features(features{i}, 'pascal', 'mean', 'vary_size_grid_search', colors(i, :));
%     legend(features(1:i), 'Location', 'BestOutside'); drawnow;
% end
% title('Specificity Prediction using SVR (Pascal dataset)[mean]');
% 
% subplot(2,2,3);
% for i=1:n_features
%     try_features(features{i}, 'pascal', 'B0', 'vary_size_grid_search', colors(i, :));
%     legend(features(1:i), 'Location', 'BestOutside'); drawnow;
% end
% title('Specificity Prediction using SVR (Pascal dataset)[B0]');
% 
% subplot(2,2,4);
% for i=1:n_features
%     try_features(features{i}, 'pascal', 'B1', 'vary_size_grid_search', colors(i, :));
%     legend(features(1:i), 'Location', 'BestOutside'); drawnow;
% end
% title('Specificity Prediction using SVR (Pascal dataset)[B1]');

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

if regexpi(features, 'attributes')
    load('../../data/memorability_mapping.mat');
    X = double(cat(2, X, Feat.anno_feats(mapping, :)));
end

if strcmpi(features, 'meanarea')
    X = double(cat(2, X, full(mean(Feat.Areas))'));
end

if regexpi(features, 'decaf')
    X = double(cat(2, X, Feat.decaf));
end

if regexpi(features, 'objectness')
    X = double(cat(2, X, Feat.objectness));
end

if regexpi(features, 'saliencymap')
    X = double(cat(2, X, Feat.saliency));
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

r_mse = zeros(length(train_size), 5); r_baseline = r_mse;

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
            
            y_const = mean(y)*ones(length(y_out), 1) + rand(length(y_out), 1);
            %y_const = mean(y)*ones(length(y_out), 1);
            
            %r_mse(i, run) = sum(abs(y_out - y(test_idx)).^2)/numel(y_out);
            %r_mse_const(i, run) = sum(abs(y_const - y(test_idx)).^2)/numel(y_const);
            r_mse(i, run) = corr(y_out, y(test_idx), 'type', 'spearman');
            r_baseline(i, run) = corr(y_const, y(test_idx), 'type', 'spearman');
            
        end
        
    end
    
end

save(sprintf('../../data/predict_specificity/%s_%s.mat', dataset, features), ...
     'r_mse', 'r_baseline', 'train_size');

end