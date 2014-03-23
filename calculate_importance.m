clear all; close all;

load('../library/cvpr_memorability_data/Data/Image data/target_features.mat', ...
    'objectnames', 'Areas', 'Counts');

load('../data/sentences_all.mat');
load('../data/memorability_mapping.mat');

min_area = 4000; % Same value as in Isola et al. NIPS memorability paper
object_pres = (full(Areas))>min_area;

n_categories = size(object_pres,1);
include = zeros(1, n_categories);

for i=1:n_categories
            
    if length(find(object_pres(i, mapping)==1))>=10
        include(i) = 1;
    end
    
end

include = find(include);

% Truncate to only images for which specificity measurements are available
object_pres = object_pres(include, mapping);
objectnames = objectnames(include);

for i=1:length(objectnames)
    fprintf('Calculating importance for object category: %s\n', objectnames{i});
    
    present_idx = find(object_pres(i, :)==1);
    
    absent_idx = find(object_pres(i, :)==0); randomorder = randperm(length(absent_idx));
    absent_idx = absent_idx(randomorder(1:length(present_idx)));
    
    similarity = zeros(length(present_idx), size(sentences,2));
    disimilarity = similarity;
    
    for j=1:length(present_idx)
        fprintf('\n');
        for k=1:size(sentences,2)
            
            clean_sentence = strrep(sentences{present_idx(j),k},'''', '''''');
            
            [~, temp] = system(sprintf('python similarity_sentence.py -s ''%s'' -o ''%s''', ...
                               clean_sentence, objectnames{i}));
            similarity(j,k) = str2double(temp);            
            fprintf('\t%s: %s (%1.2f)\n', objectnames{i}, sentences{present_idx(j), k}, similarity(j,k));
        end        
    end
    
    for j=1:length(absent_idx)
        fprintf('\n');
        for k=1:size(sentences,2)
            
            clean_sentence = strrep(sentences{absent_idx(j),k},'''', '''''');
            
            [~, temp] = system(sprintf('python similarity_sentence.py -s ''%s'' -o ''%s''', ...
                               clean_sentence, objectnames{i}));
            disimilarity(j,k) = str2double(temp);            
            fprintf('\t%s: %s (%1.2f)\n', objectnames{i}, sentences{absent_idx(j), k}, disimilarity(j,k));
        end        
    end
        
    S(i) = nanmean(similarity(:));
    D(i) = nanmean(disimilarity(:));
    
    fprintf('\nImportance for object category: %s = %0.4f\n', objectnames{i}, S(i) - D(i));
end
% Run python script


% clear all; load('../data/importance_scores.mat');
% load('../data/specificity_scores');
 
% [importance, idx] = sort(importance, 'descend');
 
 %for i=1:length(objectnames)
     %fprintf('%0.4f : %s\n', importance(i), objectnames{idx(i)});
 % end