%% write csv file

clear all; close all;

addpath('../library/cvpr_memorability_code/Code/Library/cell2csv');

fprintf('\nWriting csv file for Amazon Mechanical Turk ...');

idx = [223:888]; n_samples = length(idx);
randomorder = randperm(n_samples);
idx = idx(randomorder); % shuffle images

for i=1:n_samples/3
    mturk_csv{i+1,1} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(idx(i*3-2)) '.jpg'];
    mturk_csv{i+1,2} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(idx(i*3-1)) '.jpg'];
    mturk_csv{i+1,3} = ['https://neuro.hut.fi/~mainak/sampled_images/img_' num2str(idx(i*3)) '.jpg'];
end

mturk_csv{1,1} = 'image1_url'; mturk_csv{1,2} = 'image2_url';
mturk_csv{1,3} = 'image3_url';

cell2csv('../data/mturk/input/mturk_descriptions_batch2.csv',mturk_csv);

fprintf('[Done]');