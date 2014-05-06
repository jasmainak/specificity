clear all;

load('../../data/copy of image_search_50sentences_mud.mat');

scores_b_all = scores_b;
sent_pairs_all = sent_pairs;

clear scores_b sent_pairs;

for i=1:curr_idx
    i
    scores_b = squeeze(scores_b_all(i, :));
    sent_pairs = cell(1200, 2);
    for j=1:1200
        x1 = squeeze(sent_pairs_all(100,j,1,1,:))';
        x2 = squeeze(sent_pairs_all(100,j,2,1,:))';
        sent_pairs{j, 1} = x1;
        sent_pairs{j, 2} = x2;
    end
    save(sprintf('../../data/search_parameters/image_search_50sentences_mud_%d.mat',i), 'sent_pairs', 'scores_b');
end
