% Script to sample uniformly from the memorability data set
% and create csv file for getting MTurk descriptions

%% Specify directories
clear all; close all;

proj_dir = '/home/mainak/Desktop/specificity';
data_dir = '/library/cvpr_memorability_data/Data/Experiment data';
img_dir = '/library/cvpr_memorability_data/Data/Image data';

addpath([proj_dir '/library/cvpr_memorability_code/Code/Library/cell2csv']);

fprintf('Loading memorability images ...');

% load the data
load([proj_dir data_dir '/sorted_target_data']);
load([proj_dir img_dir '/SUN_urls.mat']);

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

%% sample images

fprintf('\nSampling and writing images ...');

% load images
load([proj_dir img_dir '/target_images.mat']);

n_samples = 222; % number of images to sample
sample_gap = floor(N/n_samples); % gap between two samples

% write n_samples images from the memorability data set
mapping = zeros(1,n_samples);
for i=1:n_samples
    I = img(:,:,:,idx((i-1)*sample_gap + 1));
    mapping(i) = idx((i-1)*sample_gap + 1);
    imwrite(I, [proj_dir '/data/sampled_images/img_' num2str(i) '.jpg']);    
end

fprintf('[Done]');

%% write csv file

fprintf('\nWriting csv file for Amazon Mechanical Turk ...');

for i=1:n_samples/3
    mturk_csv{i+1,1} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(i*3-2) '.jpg'];
    mturk_csv{i+1,2} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(i*3-1) '.jpg'];
    mturk_csv{i+1,3} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(i*3) '.jpg'];
end

mturk_csv{1,1} = 'image1_url'; mturk_csv{1,2} = 'image2_url';
mturk_csv{1,3} = 'image3_url';

cell2csv([proj_dir '/data/mturk/input/mturk_descriptions1.csv'],mturk_csv);

fprintf('[Done]');

%% Produce html file

fprintf('\nWriting html file ...');

fp = fopen([proj_dir '/data/sampled_images/index.html'],'w');
for i=1:n_samples
    fwrite(fp,['<img src="img_' num2str(i) '.jpg"><br/>']);
    fwrite(fp,sprintf('Score=%0.2f<br/>',mem(idx((i-1)*sample_gap + 1))));
end
fclose(fp);

save([proj_dir '/data/memorability_mapping.mat'],'mapping');
clearvars -except mapping img;

fprintf('[Done]\n');