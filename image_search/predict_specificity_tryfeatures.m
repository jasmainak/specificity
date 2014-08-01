% Clear variables and add paths

function predict_specificity_tryfeatures()

close all;
addpath(genpath('../../library/libsvm-3.17/'));

figure; set(gcf, 'Position', [372, 200, 1036, 800]); 

features = {'instance_occurence', 'instance_cooccurence', ...
            'instance_abslocation', 'instance_absdepth', 'decaf'};
n_features = length(features);
colors = hsv(n_features);

subplot(2,2,1);
for i=1:n_features
    try_features(features{i}, 'clipart', 'B0', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediciton using SVR (Clipart dataset)[B0]');

subplot(2, 2, 2);
for i=1:n_features
    try_features(features{i}, 'clipart', 'B1', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediciton using SVR (Clipart dataset)[B1]');

figure;
features = {'gist', 'attributes', 'decaf', 'saliencymap', 'objectness'};
n_features = length(features);
colors = hsv(n_features);

subplot(2,2,1);
for i=1:n_features
    try_features(features{i}, 'memorability', 'mean', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediction using SVR (Memorability dataset)[mean]');

features = {'gist', 'decaf', 'saliencymap', 'objectness'};
n_features = length(features);
colors = hsv(n_features);

subplot(2,2,2);
for i=1:n_features
    try_features(features{i}, 'pascal', 'mean', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediction using SVR (Pascal dataset)[mean]');

subplot(2,2,3);
for i=1:n_features
    try_features(features{i}, 'pascal', 'B0', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediction using SVR (Pascal dataset)[B0]');

subplot(2,2,4);
for i=1:n_features
    try_features(features{i}, 'pascal', 'B1', 'vary_size_grid_search', colors(i, :));
    legend(features(1:i), 'Location', 'BestOutside'); drawnow;
end
title('Specificity Prediction using SVR (Pascal dataset)[B1]');

end

function try_features(features, dataset, specificity_type, experiment, plotcolor)

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
    X = double(cat(2, full(mean(Feat.Areas))'));
end

if regexpi(features, 'decaf')
    X = double(cat(2, Feat.decaf));
end

if regexpi(features, 'objectness')
    X = double(cat(2, Feat.objectness));
end

if regexpi(features, 'saliencymap')
    X = double(cat(2, Feat.saliency));
end

if regexpi(features, 'instance_occurence')
    X = double(cat(2, Feat.instance_occurence));
end

if regexpi(features, 'instance_cooccurence')
    X = double(cat(2, Feat.instance_occurence));
end

if regexpi(features, 'instance_abslocation')
    X = double(cat(2, Feat.instance_abslocation));
end

if regexpi(features, 'instance_absdepth')
    X = double(cat(2, Feat.instance_absdepth));
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
    train_size = 50:50:400; test_idx = 401:450;
end

r_s = zeros(length(train_size), 5); r_p = r_s; r_mse = r_s;

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

            r_mse(i, run) = sum(abs(y_out - y(test_idx)).^2)/numel(y_out);
        end
        
    end
    
end

plot(gca, train_size, mean(r_mse, 2), '-d', 'color', plotcolor, ...
     'Markersize',7,'Markerfacecolor','w'); hold on;
 
xlabel('# training images','Fontsize',12);
ylabel('Mean squared error (MSE)','Fontsize',12);
set(gca,'Tickdir','out','Box','off','Fontsize',12); drawnow;

end