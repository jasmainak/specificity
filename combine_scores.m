clear all; close all;

F = load('../data/sentence_descriptions.mat');
G = load('../data/sentence_descriptions_140317.mat');

U = load('../data/specificity_scores.mat');
V = load('../data/specificity_scores_140317.mat');

sentences = cat(1, F.sentences, G.sentences);
specificity = cat(1, U.specificity, V.specificity);
scores = cat(1, U.scores, V.scores);

save('../data/sentences_all.mat', 'sentences');
save('../data/specificity_scores_all.mat', 'specificity','scores');