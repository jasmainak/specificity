% Clear variables and add paths

clearvars -except Feat; close all;

proj_dir = '..';
img_dir = [proj_dir '/library/cvpr_memorability_data/Data/Image data'];

addpath(genpath([proj_dir '/library/libsvm-3.17/']));

if ~exist('Feat','var')
    Feat = load([img_dir '/target_features.mat']);
end

load('../data/memorability_mapping.mat');
load('../data/specificity_scores.mat');
load('../library/annotations/annotations/anno_feats.mat');
load('../library/annotations/annotations/anno_names.mat');

% Curate data

f1 = full(max(Feat.Areas))'; 
f2 = Feat.gist;

%X = double(Feat.gist(mapping, :));
X = double(cat(2, Feat.gist(mapping, :), anno_feats(mapping, :)));
%X = double(cat(2, f1(mapping, :), f2(mapping, :)));
%X = double(anno_feats(mapping, :));
y = specificity;

% Do SVR

runs = 50;
folds = 5;
r = zeros(runs,folds);

for j=1:runs
    
    idx = crossvalind('Kfold',length(specificity), folds);
        
    for i=1:folds
        train_idx = (idx~=i); test_idx = (idx==i);
        
        [Z_train,mu,sigma] = zscore(X(train_idx,:));
        model = svmtrain(y(train_idx), Z_train, '-s 3');
        
        sigma0 = sigma;
        sigma0(sigma0==0) = 1;
        Z_test = bsxfun(@minus,X(test_idx,:), mu);
        Z_test = bsxfun(@rdivide, Z_test, sigma0);
        
        y_out = svmpredict(y(test_idx), Z_test, model);
        
        r(j, i) = corr(y_out, y(test_idx), 'type', 'spearman');
    end
    
end

mean(r(:))