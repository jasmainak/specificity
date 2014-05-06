%% Script to process output file from MTurk

% row_idx      -- row index for the url (1:n_HITs)
% url_idx      -- column index for the imagex_url variable (x=1,2,3)
% ans_idx      -- column index for sentence answered
% reject_idx   -- column index for rejection timestamp
% image_idx    -- image index for the sentence(1:222)
% sentence_idx -- sentence index for the image (1 x 222, 1:5)

clear all; close all;
n_images = 666; % number of images
data_collected = 222; % number of images already collected
n_sentences = 5; % number of sentences per image
n_questions = 3; % number of questions per HIT

%% Import csv files
datastruct = importdata('../data/mturk/output/Sentences_results_140317_combined.csv');
%datastruct_missing = importdata('../data/mturk/output/Sentences_results_missing.csv');

% Merge data from the two batches
%datastruct = cat(1, datastruct, datastruct_missing{2});

[headers, data] = parse_csv(datastruct);

%% Extract sentences in 2-D cell array

sentence_idx = ones(n_images,1);
sentences = cell(n_images,n_sentences);

for i=1:n_questions
    url_idx = strcmp(['Input.image' num2str(i) '_url'],headers);
    ans_idx = strcmp(['Answer.Q' num2str(i)],headers); 
    reject_idx = strcmp('RejectionTime',headers);
    
    for row_idx=1:size(data,1)
        approve = isempty(data{row_idx, reject_idx}); % use data unless rejected
        
        if approve
            url = data{row_idx, url_idx};
            image_idx = str2double(url(isstrprop(url,'digit')) ) - data_collected;
        
            datum = data{row_idx,ans_idx};
            if ~isempty(datum)
                sentences{image_idx, sentence_idx(image_idx)} = datum;
                sentence_idx(image_idx) = sentence_idx(image_idx) + 1;
            end
        end
    end
end

clearvars -except sentences;
save('../data/sentence_descriptions_140317.mat','sentences');