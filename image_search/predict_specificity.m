% Clear variables and add paths

function predict_specificity()

close all;
addpath(genpath('../../library/libsvm-3.17/'));

feat_mem = load_features('memorability');

figure; set(gcf, 'Position', [372, 200, 1036, 800]); 

% features = {'gist', 'attributes', 'decaf', 'saliencymap', 'objectness'};
features = {'saliencymap'};
n_features = length(features);
colors = hsv(n_features);

%subplot(2,1,1);
for i=1:n_features
    try_features(feat_mem, features{i}, 'memorability', 'vary_size', colors(i, :));
    legend(reshape([features(1:i); features(1:i)], 1, i*2), 'Location', ...
           'BestOutside'); drawnow;
end
title('Specificity Prediction using SVR (Memorability dataset)');
% 
% feat_pascal = load_features('pascal');
% features = {'gist', 'decaf', 'saliencymap', 'objectness'};
% n_features = length(features);
% colors = hsv(n_features);
% 
% subplot(2,1,2);
% for i=1:n_features
%     try_features(feat_pascal, features{i}, 'pascal', 'vary_size', colors(i, :));
%     legend(reshape([features(1:i); features(1:i)], 1, i*2), 'Location', ...
%            'BestOutside'); drawnow;
% end
% title('Specificity Prediction using SVR (Pascal dataset)');

end

function Feat = load_features(dataset)

fprintf('\nLoading %s dataset ... ', dataset);

if strcmpi(dataset, 'memorability')
    load('../../data/image_features/feat_mem.mat');
elseif strcmpi(dataset, 'pascal')
    load('../../data/image_features/feat_pascal.mat');
end
% addpath('../aux_functions/');
% 

% 
% if strcmpi(dataset, 'memorability')
%     
%     img_dir = '../../library/cvpr_memorability_data/Data/Image data';
%     Feat = load([img_dir '/target_features.mat']);
%     load('../../library/annotations/annotations/anno_feats.mat');
%     Feat.anno_feats = anno_feats;
% 
% end
% 
% if strcmpi(dataset, 'pascal')
%     load('../../data/sentences/pascal_1000_img_50_sent.mat', 'pascal_urls');
%     urls = pascal_urls;
% elseif strcmpi(dataset, 'memorability')
%     load('../../data/sentences/memorability_888_img_5_sent.mat', 'memorability_urls');
%     urls = memorability_urls;
% end
% 
% for i=1:length(urls)
%     progressbar(i, 5, length(urls));
% 
%     filename = strsplit(urls{i}, '/');
% 
%     mat = load(sprintf('../../data/image_features/decaf/%s/%s_decaf.mat', dataset, cell2mat(filename(end))));
%     Feat.decaf(i, :) = double(mat.fc6n);
%         
%     mat = load(sprintf('../../data/image_features/objectness/%s/%s_objectness.mat', dataset, cell2mat(filename(end))));
%     heatmap = imresize(mat.obj_heatmap, [96, 96]);
%     Feat.objectness(i, :) = double(heatmap(:));
%     
%     mat = load(sprintf('../../data/image_features/saliencymap/%s/%s_saliencymap.mat', dataset, cell2mat(filename(end))));
%     saliency = imresize(mat.saliencyMap, [96, 96]);
%     Feat.saliency(i, :) = saliency(:);
%     
%     if strcmpi(dataset, 'pascal')
%         mat = load(sprintf('../../data/image_features/gist/%s/%s_gist.mat', dataset, cell2mat(filename(end))));
%         Feat.gist(i, :) = mat.gist_features(:);
%     end
%     
% end

fprintf(' [Done]');

end

function try_features(Feat, features, dataset, experiment, plotcolor)

% Labels for prediction

if strcmpi(dataset, 'memorability')
    load('../../data/memorability_mapping.mat');
    load('../../data/specificity_automated.mat', 'specificity_w');
    y = specificity_w;
elseif strcmpi(dataset, 'pascal')
    load('../../data/image_search_50sentences_parameters.mat', 'scores_w');
    y = mean(scores_w, 2);
end
    
% Feature Vector

X = [];

if regexpi(features, 'gist')
    if strcmpi(dataset, 'memorability')
        X = cat(2, X, double(Feat.gist(mapping, :)));
    elseif strcmpi(dataset, 'pascal')
        X = cat(2, X, double(Feat.gist));
    end
end

if regexpi(features, 'attributes')
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

if strcmpi(dataset, 'pascal')
    y = y(~isnan(y));
    X = X(~isnan(y), :);
end

% Grid Search

folds = 5;
idx = crossvalind('Kfold',length(y), folds);

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

% Vary size of training data

if strcmpi(dataset, 'memorability')
    train_size = 100:100:700; test_idx = 701:888;
elseif strcmpi(dataset, 'pascal')
    train_size = 100:100:800; test_idx = 801:length(y);
end

r_s = zeros(length(train_size), 5); r_p = r_s; r_mse = r_s;

if strcmpi(experiment, 'vary_size')
    
    % Grid search to select C and gamma
    Z_train = zscore(X(1:train_size(end), :));
    
    optimalg = 1/size(Z_train,2); % 1/number of features
    bestcv = Inf;
    for log10C=-1:3
        for g = optimalg/2:optimalg/10:optimalg*1.5
            cv = svmtrain2(y(1:train_size(end)), Z_train, ['-s 3 -v 5 -q -c ' num2str(10^log10C) ' -g ', num2str(g)]);
            if (cv < bestcv),
                bestcv = cv; bestc = 10^log10C; bestg = g;
            end
            fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)\n', 10^log10C, g, cv, bestc, bestg, bestcv);
        end
    end

    % Refine grid search
    for C=bestc/2:bestc/10:bestc*1.5
        for g = optimalg/2:optimalg/10:optimalg*1.5
            cv = svmtrain2(y(1:train_size(end)), Z_train, ['-s 3 -v 5 -q -c ' num2str(C) ' -g ', num2str(g)]);
            if (cv < bestcv),
                bestcv = cv; bestc = C; bestg = g;
            end
            fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)\n', C, g, cv, bestc, bestg, bestcv);
        end
    end

    for run=1:50
        
        randomorder = randperm(length(y));
        X = X(randomorder, :);
        y = y(randomorder);
        
        for i=1:length(train_size)
            
            fprintf('\nFeature = %s, Run %d, Trainsize = %d\n', features, run, train_size(i));

            train_idx = 1:train_size(i); 
            test_idx = train_idx; % uncomment later
            
            optimalc = bestc*train_size(i)/train_size(end);
            
            [Z_train,mu,sigma] = zscore(X(train_idx,:));
            
            model = svmtrain2(y(train_idx), Z_train, ['-s 3 -c ' num2str(optimalc), ' g ' num2str(bestg)]);
            
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            y_out = svmpredict2(y(test_idx), Z_test, model, '-q');
            
            %r_s(i, run) = corr(y_out, y(test_idx), 'type', 'spearman');
            %r_p(i, run) = corr(y_out, y(test_idx), 'type', 'pearson');
            r_mse(i, run) = sum(abs(y_out - y(test_idx)).^2)/numel(y_out);
        end
        
    end
    
end

%plot(gca, train_size, mean(r_s,2), '-o', 'color', plotcolor, ...
%     'Markersize',7,'Markerfacecolor','w'); hold on;
%plot(gca, train_size, mean(r_p,2), '-s', 'color', plotcolor, ...
%     'Markersize',7,'Markerfacecolor','w'); hold on;
plot(gca, train_size, mean(r_mse, 2), '-d', 'color', plotcolor, ...
     'Markersize',7,'Markerfacecolor','w'); hold on;
 
xlabel('# training images','Fontsize',12);
ylabel('Correlation (spearman / pearson)','Fontsize',12);
set(gca,'Tickdir','out','Box','off','Fontsize',12); drawnow;

end