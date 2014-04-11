clear all; close all;

load('../data/pascal_1000_img_50_sent.mat');
load('../data/clipart_500_img_48_sent.mat');

pascal_sentences = cell(1000,50); pascal_urls = cell(1000,1);
for i=1:length(train_sent_final)
    for j=1:50
        pascal_sentences{i,j} = train_sent_final(i).sentences{j};
        pascal_urls{i} = train_sent_final(i).name;
    end
end

clipart_sentences = cell(500,48); clipart_urls = cell(500,1);
for i=1:length(test_clip)
    for j=1:48
        clipart_sentences{i,j} = test_clip(i).sentences{j};
        clipart_urls{i} = test_clip(i).name;
    end
end

save('../data/pascal_1000_img_50_sent.mat','train_sent_final',...
     'pascal_sentences','pascal_urls');