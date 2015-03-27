% Unsuccessful attempt to try finding image from which sentence came
% by searching through all images. The problem is that many images can
% have the same sentence leading to clashes. It is better to redo
% the sentence similarities cleanly.

clear all;

load('../../data/sentences/pascal_1000_img_50_sent.mat');

count = 0;
for i=1:1%1000
    load(sprintf('../../data/search_parameters/pascal/mu_d/image_search_50sentences_mud_%d.mat', i - 1), 'sent_pairs');
    im_idxs = zeros(length(sent_pairs), 1);
    for j=1:length(sent_pairs)
        [im_idx, sent_idx] = ind2sub(size(pascal_sentences), find(strcmp(sent_pairs{j, 2}, pascal_sentences)));
        fprintf('%s ... %s\n', sent_pairs{j, 2}, pascal_sentences{im_idx, sent_idx});
        
        if length(unique(im_idx)) > 1
            im_idx(j) = im_idx(1);
            count = count + 1;
        else
            im_idxs(j) = unique(im_idx);  % Two sentences may be the same for a given image, which is ok
        end
    end
end