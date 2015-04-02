% CALCULATE_PERCENTAGE_CURVE finds the percentage of times baseline is
% beaten by specificity
%
% AUTHOR: Mainak Jas
%
% See also: plot_percentage_curve.py
function calculate_percentage_curve()

    [clipart.stats_b, clipart.stats_s, clipart.stats_gt] = do_search('clipart');
    [pascal.stats_b, pascal.stats_s, pascal.stats_gt] = do_search('pascal');

    save('../../data/search_results/percentage_results.mat', 'pascal', 'clipart');
    
end

function [stats_b, stats_s, stats_gt] = do_search(dataset)

    addpath(genpath('../aux_functions/'));
    addpath('../io/');
    addpath('utils/');

    % Load image features, query-ref similarities and predicted LR
    % parameters
    [~, ~, s, ~, ~, urls, ~] = load_search_parameters(dataset);
    load(['../../data/image_search/' dataset '/LR_params/Pred/predicted_LR.mat']);
    split_url = strsplit(urls{1}, '/');
    filename = split_url{end};
    load(['../../data/image_search/' dataset '/LR_params/GT/predicted_img_' filename '.mat']);
    y = B(:, 1); z = B(:, 2);

    n_images = length(s);
    
    % Calculate image search results
    rank_b = baseline_search(s);
    rank_s = specificity_search(s, y_pred, z_pred);
    rank_gt = specificity_search(s, y, z);

    [~, stats_b] = calculate_percentage(rank_b, rank_b, n_images);
    [~, stats_gt] = calculate_percentage(rank_gt, rank_b, n_images);
    [~, stats_s] = calculate_percentage(rank_s, rank_b, n_images);
 
end

function [y, stats] = calculate_percentage(rank, rank_b, n_images)

    y = zeros(n_images, 2);

    for k=1:n_images
        y(k, 1) = length(find(rank_b - rank >= k & rank_b > rank))/n_images*100;
    end

    for k=1:n_images
        y(k, 2) = length(find(rank - rank_b >= k & rank > rank_b))/n_images*100;
    end

    stats.lt_b = length(find(rank<rank_b))/length(rank_b)*100;
    stats.gt_b = length(find(rank>rank_b))/length(rank_b)*100;
    stats.eq_b = length(find(rank==rank_b))/length(rank_b)*100;
    stats.y = y;
    stats.rank = mean(rank);
    stats.target_ranks = rank;
    stats.rank_b = rank_b;

end
