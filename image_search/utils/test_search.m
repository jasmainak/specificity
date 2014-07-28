clear all; close all;
load('../../../data/specificity_alldatasets.mat');

eval(['y = specificity.pascal.B0;']);
eval(['z = specificity.pascal.B1;']);

cd('../../io/'); % for load_search_parameters
[~, ~, s, ~, ~, ~] = load_search_parameters('pascal');
cd('../image_search/utils/');

rank_s = specificity_search(s, y, z);
rank_b = baseline_search(s);

save('test_search.mat', 'y', 'z', 's', 'rank_s', 'rank_b');
