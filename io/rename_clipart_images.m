load('../../data/sentences/clipart_450_img_48_sent.mat', 'clipart_urls');
S = load('../../data/sentences/clipart_500_img_48_sent.mat');

for i=1:50
    temp = strsplit(clipart_urls{i}, '/');
    clipart_urls{i} = cell2mat(temp(end));

    temp = strsplit(S.clipart_urls{i}, '/');
    S.clipart_urls{i} = cell2mat(temp(end));
    
    load(sprintf('../../data/image_features/decaf/clipart/%s_decaf.mat', clipart_urls{i}));
    save(sprintf('../../data/image_features/decaf/clipart/%s_decaf.mat', S.clipart_urls{i}), ...
         'fc6', 'fc6n', 'fc7', 'fc7n');
end