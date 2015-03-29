% CALCULATE_PERCENTAGE_CURVE finds the percentage of times baseline is
% beaten by specificity
%
% INPUT
%   data/search_parameters/search_parameters_{dataset}.mat
%   data/search_parameters/{dataset}/predicted_LR.mat
%   data/specificity_alldatasets.mat
%
% AUTHOR: Mainak Jas
%
% See also: plot_percentage_curve.py
function calculate_percentage_curve()

    [pascal.stats_b, pascal.stats_s, pascal.stats_min, pascal.stats_gt] = do_search('pascal');
    [clipart.stats.b, clipart.stats_s, clipart.stats_min, clipart.stats_gt] = do_search('clipart');

    save('../../data/search_results/percentage_results.mat', 'pascal', 'clipart');

    
end

function [stats_b, stats_s, stats_min, stats_gt] = do_search(dataset)

    addpath(genpath('../aux_functions/'));
    addpath('utils/');

    % Load image features, query-ref similarities and predicted LR
    % parameters
    load('../../data/image_features/feat_pascal.mat'); % remove this line
    load(['../../data/search_parameters/search_parameters_' dataset '.mat'], 's');
    load(['../../data/search_parameters/' dataset '/predicted_LR.mat']);
    
    feat = Feat.decaf;
    n_images = length(s);

    [rank_min, rank_s, rank_b] = predict_best_method(s, y_pred, z_pred, feat);
    
    load('../../data/specificity_alldatasets.mat');
    eval(['y = specificity.' dataset '.B0;']);
    eval(['z = specificity.' dataset '.B1;']);
    rank_gt = specificity_search(s, y, z);
   
    rank_oracle = min([rank_min; rank_s; rank_b]);

    [~, stats_b] = calculate_percentage(rank_b, rank_b, n_images);
    [~, stats_gt] = calculate_percentage(rank_gt, rank_b, n_images);
    [~, stats_s] = calculate_percentage(rank_s, rank_b, n_images);
    [~, stats_min] = calculate_percentage(rank_min, rank_b, n_images);

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
