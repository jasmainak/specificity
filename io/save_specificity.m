function save_specificity()

specificity.pascal = find_specificity('pascal');
specificity.clipart = find_specificity('clipart');

load('../../data/specificity_automated.mat', 'specificity_w');
specificity.memorability.mean = specificity_w;

save('../../data/specificity_alldatasets.mat', 'specificity');

end

function specificity = find_specificity(dataset)

[scores_b, scores_w, ~, sentences, ~, ~] = load_search_parameters(dataset);
[n_images, ~] = size(sentences);

fprintf('Finding specificity for dataset %s ... ', dataset);
for idx=1:n_images

    progressbar(idx, 10, n_images);

    y_s = scores_w(idx,:);
    y_d = scores_b(idx,:);

    len = min(length(y_s), length(y_d));

    X = cat(2, y_s(1:len), y_d(1:len));
    labels = cat(1, ones(len,1), zeros(len,1));

    B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
end
clear X;

specificity.B0 = B(:, 1);
specificity.B1 = B(:, 2);
specificity.mean = mean(scores_w, 2);

fprintf(' [Done]\n');

end