function plot_percentage_curve()

    addpath(genpath('../aux_functions/'));
    addpath('utils/');

    load('../../data/image_features/feat_pascal.mat');
    load('search_parameters_pascal.mat', 's');
    load('../../data/search_parameters/pascal/predicted_LR.mat');

    feat = Feat.decaf;
    n_images = length(s);

    [rank_best, rank_min, rank_s, rank_b] = predict_best_method(s, y_pred, z_pred, feat);
    rank_oracle = min([rank_min; rank_s; rank_b]);

    y_b = calculate_percentage(rank_b, rank_b, n_images);
    [y_s, stats_s] = calculate_percentage(rank_s, rank_b, n_images);
    [y_min, stats_min] = calculate_percentage(rank_min, rank_b, n_images);
    [y_best, stats_best] = calculate_percentage(rank_best, rank_b, n_images);
    [y_oracle, stats_oracle] = calculate_percentage(rank_oracle, rank_b, n_images);

    plot(1:n_images, y_b); hold on;
    %plot(1:n_images, y_s, 'k'); plot(1:n_images, y_min, 'g');
    plot(1:n_images, y_best, 'r');

    xlabel('K', 'Fontsize', 12);
    ylabel('% of target images with rank \leq K');
    title('Retrieval curve', 'Fontsize', 14);
    legend('Baseline', 'Pick best', 'Location', 'SouthEast');
    set(gca, 'Box', 'off', 'tickdir', 'out');

    %magnifyOnFigure(gcf, 'initialPositionSecondaryAxes', [229.6 214.46 130.2 102.54], ...
    %                'initialPositionMagnifier', [139.742 332.208 76.1169 49.0446]);
    %magnifyOnFigure(gcf, 'initialPositionSecondaryAxes', [228.6 79.46 130.2 102.54], ...
    %                'initialPositionMagnifier', [88.8 309.03 43.4 44.1401]);
    magnifyOnFigure(gcf, 'initialPositionSecondaryAxes', [235.6 179.46 130.2 102.54], ...
                    'initialPositionMagnifier', [97.8087 331.437 108.096 44.1401]);
end

function [y, stats] = calculate_percentage(rank, rank_b, n_images)

    y = zeros(n_images, 1);
    for k=1:n_images
        y(k) = length(find(rank<=k))/n_images*100;
    end

    stats.lt_b = length(find(rank<rank_b))/length(rank_b)*100;
    stats.gt_b = length(find(rank>rank_b))/length(rank_b)*100;
    stats.eq_b = length(find(rank==rank_b))/length(rank_b)*100;

end
