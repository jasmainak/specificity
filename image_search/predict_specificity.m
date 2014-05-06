% Clear variables and add paths

clearvars -except Feat; close all;
addpath(genpath('../../library/libsvm-3.17/'));

features = 'gist923'; % {'gist', 'gist923', '923', 'gistarea'} % only for memorability dataset
experiment = 'vary_size'; % {'grid_search', 'vary_size'}
dataset = 'pascal'; % {'pascal', 'memorability'}

if strcmpi(dataset, 'memorability')
    img_dir = '../../library/cvpr_memorability_data/Data/Image data';
    
    if ~exist('Feat','var')
        Feat = load([img_dir '/target_features.mat']);
    end
    
    load('../../data/memorability_mapping.mat');
    load('../../data/specificity_scores_all.mat');
    load('../../library/annotations/annotations/anno_feats.mat');
    load('../../library/annotations/annotations/anno_names.mat');
    
    % Curate data
    
    if strcmpi(features, 'gist')
        X = double(Feat.gist(mapping, :));
    elseif strcmpi(features, 'gist923')
        X = double(cat(2, Feat.gist(mapping, :), anno_feats(mapping, :)));
    elseif strcmpi(features, '923')
        f1 = full(max(Feat.Areas))';
        f2 = Feat.gist;
        X = double(cat(2, f1(mapping, :), f2(mapping, :)));
    elseif strcmpi(features, 'gistarea')
        X = double(anno_feats(mapping, :));
    end
        
elseif strcmpi(dataset, 'pascal')
    load('../../data/pascal_1000_img_50_sent.mat', 'pascal_urls');
    load('../../data/image_search_50sentences_parameters.mat', 'scores_w');
    
    specificity = nanmean(scores_w, 2); %specificity score
   
    for i=1:length(pascal_urls)
        filename = strsplit(pascal_urls{i}, '/');
        load(sprintf('../../data/pascal_decaf/%s_decaf.mat',cell2mat(filename(end))), 'fc6');
        X(i, :) = double(fc6); clear fc6n;
    end
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
    
    mean(r, 3)
    
end

if strcmpi(experiment, 'vary_size')    
    
    train_size = [100:100:700];
    r = zeros(length(train_size), 5);
    
    for j=1:5
        
        randomorder = randperm(length(specificity));
        X = X(randomorder, :);
        y = y(randomorder);
        
        for i=1:length(train_size)
            
            train_idx = 1:train_size(i); test_idx = 701:888;
            
            [Z_train,mu,sigma] = zscore(X(train_idx,:));
            
            model = svmtrain(y(train_idx), Z_train, '-s 3');
            
            sigma0 = sigma;
            sigma0(sigma0==0) = 1;
            Z_test = bsxfun(@minus,X(test_idx,:), mu);
            Z_test = bsxfun(@rdivide, Z_test, sigma0);
            
            y_out = svmpredict(y(test_idx), Z_test, model);
            
            r(i, j) = corr(y_out, y(test_idx), 'type', 'spearman');
        end
        
    end
    
    plot(train_size, mean(r,2)); hold on;
    plot(train_size, mean(r,2), 'bo', 'Markersize',6,'Markerfacecolor','w');
    xlabel('No. of training images','Fontsize',12);
    ylabel('Spearman''s correlation','Fontsize',12);
    set(gca,'Tickdir','out','Box','off','Fontsize',12);
    
end