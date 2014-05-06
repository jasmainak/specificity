% Script to calculate specificity score of each image
clear all; close all;

%% Read CSV File
%datastruct = importdata('../data/mturk/output/sentence_similarity_results.csv');
datastruct = importdata('../data/mturk/output/sentence_similarity_results_140317_cleaned.csv');
%load('../data/sentence_descriptions.mat');
load('../data/sentence_descriptions_140317.mat');

[headers, data] = parse_csv(datastruct);

n_images = 666; n_sentences = 5; n_assignments = 3;
n_questions = 6; % number of questions per HIT
data_collected = 222;

scores = zeros(n_images, nchoosek(n_sentences,2), n_assignments);
w = ones(n_images, nchoosek(n_sentences,2));

combinations = {'12', '13', '14', '15', '23', '24', '25', '34', '35', '45'};

for i=1:n_questions
    img_idx = strcmp(['Input.im' num2str(i)], headers);
    s1_idx = strcmp(['Input.im' num2str(i) '_s1'], headers);
    s2_idx = strcmp(['Input.im' num2str(i) '_s2'], headers);
    rate_idx = strcmp(['Answer.Q' num2str(i)], headers);
    
    reject_idx = strcmp('RejectionTime',headers);
    
    for row_idx=1:size(data,1)
        approve = isempty(data{row_idx, reject_idx}); % use data unless rejected
        
        if approve
            u = str2double(data{row_idx, img_idx});
            v = find(strcmp([data{row_idx, s1_idx} data{row_idx, s2_idx}], ...
                     combinations) == 1);
            scores(u,v,w(u,v)) = str2double(data{row_idx, rate_idx});
            w(u,v) = w(u,v) + 1;
        end
    end
end

clearvars -except scores w;

scores = scores(:,:,1:3); % Take only 3 assignments per pair
scores(scores==0) = NaN; % Ignore scores not available

scores = (scores - 1)/(10 - 1); % map scores to range [0, 1]
specificity = nanmean(nanmean(scores,3),2);

save('../data/specificity_scores_140317.mat','specificity','scores');