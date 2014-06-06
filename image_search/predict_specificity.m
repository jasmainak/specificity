% Clear variables and add paths

function predict_specificity()
clear all; close all;
addpath(genpath('../../library/libsvm-3.17/'));

img_dir = '../../library/cvpr_memorability_data/Data/Image data';
Feat = load([img_dir '/target_features.mat']);

figure;
h1 = try_features(Feat, 'gist', 'vary_size', 'r');
h2 = try_features(Feat, 'gist-meanarea', 'vary_size', 'g');
h3 = try_features(Feat, 'attributes', 'vary_size', 'b');
h4 = try_features(Feat, 'attributes-meanarea', 'vary_size', 'k');
h5 = try_features(Feat, 'gist-attributes', 'vary_size', 'm');
h6 = try_features(Feat, 'gist-attributes-meanarea', 'vary_size', 'y');

title('Specificity Prediction using SVR');
legend([h1, h2, h3, h4, h5, h6], 'gist', 'gistarea', 'attributes', ...
       'attributesarea', 'gistattributes', 'gistattributesarea', ...
       'Location', 'BestOutside');

end

function h = try_features(Feat, features, experiment, plotcolor)

load('../../data/memorability_mapping.mat');
load('../../data/specificity_scores_all.mat');
load('../../library/annotations/annotations/anno_feats.mat');
load('../../library/annotations/annotations/anno_names.mat');

% Curate data

X = [];
if regexpi(features, 'gist')
    X = cat(2, X, double(Feat.gist(mapping, :)));
end

if regexpi(features, 'attributes')
    X = double(cat(2, X, anno_feats(mapping, :)));
end

if strcmpi(features, 'meanarea')
    X = double(cat(2, full(mean(Feat.Areas))'));
end

y = specificity;

% Do SVR

folds = 5;
idx = crossvalind('Kfold',length(specificity), folds);

if strcmpi(experiment, 'grid_search')
    
    gamma = [0.01, 0.001, 0.0001, 0.00001, 0.000001];
    C = [1, 10, 100, 1000];
    
    r = zeros(length(gamma), length(C), folds);
    
    for k=1:length(gamma)
        for j=1:length(C)
            for i=1:folds
                train_idx = (idx~=i); test_idx = (idx==i);
                
                [Z_train,mu,sigma] = zscore(X(train_idx,:));
                
                model = svmtrain(y(train_idx), Z_train, sprintf('-s 3 -c %d -g %f', C(j), gamma(k)));
                
                sigma0 = sigma;
                sigma0(sigma0==0) = 1;
                Z_test = bsxfun(@minus,X(test_idx,:), mu);
                Z_test = bsxfun(@rdivide, Z_test, sigma0);
                
                y_out = svmpredict(y(test_idx), Z_test, model);
                
                r(k, j, i) = corr(y_out, y(test_idx), 'type', 'spearman');
            end
        end
    end
    
    mean(r, 3);
    
end

train_size = [100:100:700];
r = zeros(length(train_size), 5);

if strcmpi(experiment, 'vary_size')
    
    for j=1:10
        
        randomorder = randperm(length(specificity));
        X = X(randomorder, :);
        y = y(randomorder);
        
        for i=1:length(train_size)
            
            fprintf('Feature = %s, Subsample %d, Trainsize = %d\n', features, j, train_size(i));

            train_idx = 1:train_size(i); test_idx = 701:888;
            
            [Z_train,mu,sigma] = zscore(X(train_idx,:));
            
            model = svmtrain(y(train_idx), Z_train, '-s 3 -q');
            
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            y_out = svmpredict(y(test_idx), Z_test, model, '-q');
            
            r(i, j) = corr(y_out, y(test_idx), 'type', 'spearman');
        end
        
    end
    
end

h = plot(gca, train_size, mean(r,2), plotcolor); hold on;
plot(gca, train_size, mean(r,2), [plotcolor, 'o'], 'Markersize',7,'Markerfacecolor','w');
xlabel('No. of training images','Fontsize',12);
ylabel('Spearman''s correlation','Fontsize',12);
set(gca,'Tickdir','out','Box','off','Fontsize',12); drawnow;

end