% Script to sample uniformly from the memorability data set
% and create csv file for getting MTurk descriptions

%% Specify directories
clear all; close all;
data_dir = '../library/cvpr_memorability_data/Data/Experiment data';

% load the data
fprintf('Loading memorability images ...');
load([data_dir '/sorted_target_data']);
fprintf('[Done]');

%% calculate memorability and sort it

% extract hits and misses
N = 2222;
hits = zeros(N,1);
misses = zeros(N,1);
for i=1:N
    hits(i) =  sorted_target_data{i}.hits;
    misses(i) = sorted_target_data{i}.misses;
end

mem = hits./(hits+misses);
[~, idx] = sort(mem, 'descend');

save('../data/memorability_sorted.mat','mem', 'idx');
