clear all; close all;

%% sample images
img_dir = '../library/cvpr_memorability_data/Data/Image data';

% load images
fprintf('\nLoading images ... ');
load([img_dir '/target_images.mat']);
fprintf('\n[Done]');

load('../data/memorability_sorted.mat');

% start_idx = 1;
start_idx = [3,6,8];
data_collected = 222;

n_samples = 222; % 1/10th of the data set
sample_gap = 10; % gap between two samples

fprintf('\nWriting images ... ');

% write n_samples images from the memorability data set
load('../data/memorability_mapping.mat','mapping');
mapping = cat(2, mapping(1:data_collected), ...
              zeros(1,n_samples*length(start_idx)));

for j=1:length(start_idx)
    for i=1:n_samples
        img_idx = data_collected + (j-1)*n_samples + i;
        
        I = img(:,:,:,idx((i-1)*sample_gap + start_idx(j)));
        mapping(img_idx) = idx((i-1)*sample_gap + start_idx(j));
        imwrite(I, ['../data/sampled_images/img_' num2str(img_idx) '.jpg']);
    end
end

fprintf('[Done]');

%% Produce html file

fprintf('\nWriting html file ...');

n_images = length(dir('../data/sampled_images/*.jpg'));

fp = fopen('../data/sampled_images/index.html','w');
for i=1:n_images
    fwrite(fp,['<img src="img_' num2str(i) '.jpg"><br/>']);
    fwrite(fp,sprintf('Score=%0.2f<br/>',mem(mapping(i))));
end
fclose(fp);

save('../data/memorability_mapping.mat','mapping', 'mem');
%save('../data/memorability_mapping_intersubject', 'mapping', 'mem');
clearvars -except mapping img;

fprintf('[Done]\n');