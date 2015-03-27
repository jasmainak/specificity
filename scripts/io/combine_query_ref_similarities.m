clear all;

s_dir = '../../data/search_parameters/pascal/s/';
folders = dir(s_dir);
load('../../data/sentences/pascal_1000_img_50_sent.mat', 'pascal_urls');

u = 1;
for i=3:length(folders)
    fprintf('\n%s', folders(i).name);

    % skip if all 1000 pairs are not available
    files = dir([s_dir folders(i).name]);
    if length(files) ~= 1002
        fprintf('[skipping]');
        continue;
    end

    query_sentences = cell(1000, 1);
    s_pair = zeros(1000, 1000);

    for query_idx=1:length(pascal_urls)
        split_url = strsplit(pascal_urls{query_idx}, '/');
        filename = [split_url{end} '.mat'];
        load([s_dir folders(i).name '/target_' filename]);
        query_sentences{query_idx} = query_sentence;
        s_pair(query_idx, :) = s;
    end

    similarity(u).s = s_pair;
    similarity(u).pair = folders(i).name;
    similarity(u).ref_sentences = ref_sentences;
    similarity(u).query_sentences = query_sentences;
    similarity(u).target_images = pascal_urls;
    u = u + 1;
end

save([s_dir 'combined_similarity.mat'], 'similarity');