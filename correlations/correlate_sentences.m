clear all; close all;

load('../../data/sentences/memorability_sent_lengths.mat');
load('../../data/specificity_scores_all.mat');

sent_lengths = double(cell2mat(sent_lengths));

[crMeanLength, pval1] = corr(specificity, mean(sent_lengths, 2), 'type', 'spearman');
[crVaryLength, pval2] = corr(specificity, std(sent_lengths, 0, 2), 'type', 'spearman');