clear all; close all;

%% Read CSV File
datastruct = importdata('specificity_data/sentence_similarity_test_results.csv');
load sentence_descriptions.mat;

[headers, data] = parse_csv(datastruct);

n_images = 222; n_sentences = 5; n_assignments = 3;
n_questions = 3; % number of questions per HIT

scores = zeros(n_images, nchoosek(n_sentences,2), n_assignments);
combinations = {'12', '13', '14', '15', '23', '24', '25', '34', '35', '45'};