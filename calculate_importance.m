clear all; close all;

load('../library/cvpr_memorability_data/Data/Image data/target_features.mat', ...
    'objectnames', 'Areas', 'Counts');
load('../data/sentences_all.mat');
load('../data/memorability_mapping.mat');
load('../data/specificity_scores_all.mat');

platform = 'python'; % 'python' / 'matlab'

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

save('../data/object_presence.mat','objectnames','object_pres');

if strcmpi(platform,'matlab')
    
    for i=1:length(objectnames)
        fprintf(fp, 'Calculating importance for object category: %s\n', objectnames{i});
        
        present_idx = find(object_pres(i, :)==1);
        
        absent_idx = find(object_pres(i, :)==0); randomorder = randperm(length(absent_idx));
        absent_idx = absent_idx(randomorder(1:length(present_idx)));
        
        similarity = zeros(length(present_idx), size(sentences,2));
        disimilarity = similarity;
        
        fprintf(fp, '\nSIMILARITY\n');
        
        for j=1:length(present_idx)
            fprintf(fp, '\n');
            for k=1:size(sentences,2)
                
                clean_sentence = strrep(sentences{present_idx(j),k},'''', '''''');
                
                [~, temp] = system(sprintf('python similarity_sentence.py -s ''%s'' -o ''%s''', ...
                    clean_sentence, objectnames{i}));
                similarity(j,k) = str2double(temp);
                fprintf(fp, '\t%s: %s (%1.2f)\n', objectnames{i}, sentences{present_idx(j), k}, similarity(j,k));
            end
        end
        
        fprintf(fp, '\nDISIMILARITY\n');
        
        for j=1:length(absent_idx)
            fprintf(fp, '\n');
            for k=1:size(sentences,2)
                
                clean_sentence = strrep(sentences{absent_idx(j),k},'''', '''''');
                
                [~, temp] = system(sprintf('python similarity_sentence.py -s ''%s'' -o ''%s''', ...
                    clean_sentence, objectnames{i}));
                disimilarity(j,k) = str2double(temp);
                fprintf(fp, '\t%s: %s (%1.2f)\n', objectnames{i}, sentences{absent_idx(j), k}, disimilarity(j,k));
            end
        end
        
        S(i) = nanmean(similarity(:));
        D(i) = nanmean(disimilarity(:));
        importance(i) = S(i) - D(i);
        
        fprintf(fp, '\nImportance for object category: %s = %0.4f (%0.4f - %0.4f)\n', ...
            objectnames{i}, S(i) - D(i), S(i), D(i));
    end
    
else
    
    % system('python calculate_importance.py');
        
end

load('../data/importance_scores.mat');

% Remove 'person sitting' category
object_pres = object_pres([1:22, 24:41], :);
objectnames = objectnames([1:22, 24:41]);
importance = importance([1:22, 24:41]);
S = S([1:22, 24:41]); D = D([1:22, 24:41]);

% Correlation of important images with specificity
for i=1:length(specificity)
    obj_idx = find(object_pres(:, i)==1);
    if ~isempty(importance(obj_idx))
        img_score(i) = median(importance(obj_idx));
    else
        img_score(i) = 0;
    end
end

corr(img_score', specificity, 'type', 'spearman')

clear evaluation;

% Correlation of importance with mean specificity of that object
for i=1:length(importance)    
    im_idx = find(object_pres(i, :)==1);
    obj_score(i) = mean(specificity(im_idx));
end

corr(obj_score', importance, 'type', 'spearman')

fp = fopen('../qualitative/importance_output.txt','a');

[~, idx] = sort(importance, 'descend');

fprintf(fp, '\nImportance sorted in descending order\n');

for i=1:length(objectnames)
    fprintf(fp, '%0.4f (%0.4f - %0.4f) : %s\n', importance(idx(i)), S(idx(i)), D(idx(i)), objectnames{idx(i)});
end

[~, idx] = sort(S, 'descend');

fprintf(fp, '\nImportance sorted using only S\n');

for i=1:length(objectnames)
    fprintf(fp, '%0.4f : %s\n', S(idx(i)), objectnames{idx(i)});
end

fclose(fp);