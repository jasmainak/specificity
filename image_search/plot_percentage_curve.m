clear all; close all;
addpath('../../library/export_fig/');
load('../../data/search_results/percentage_results.mat');

plot(pascal.stats_gt.y(:, 1), 'b-o', 'Linewidth', 2, 'MarkerSize', 7, 'MarkerFacecolor', 'w'); hold on;
plot(clipart.stats_gt.y(:,1), 'r-s', 'Linewidth', 2, 'MarkerSize', 7, 'MarkerFacecolor', 'w');
plot(pascal.stats_s.y(:, 1), 'b--o', 'Linewidth', 2, 'MarkerSize', 7, 'MarkerFacecolor', 'w');
plot(clipart.stats_s.y(:, 1), 'r--s', 'Linewidth', 2, 'MarkerSize', 7, 'MarkerFacecolor', 'w');
%plot(pascal.stats_min.y(:, 1), 'r', 'Linewidth', 2);
%plot(clipart.stats_min.y(:, 1), 'r--', 'Linewidth', 2);

%legend('Specificity', 'Ground truth', 'Min-rank', 'pick best', 'Location', 'NorthEast');
legend('GT-Spec (PASCAL-50S)', 'GT-Spec (ABSTRACT-50S)', ...
    'P-Spec (PASCAL-50S)', 'P-Spec (ABSTRACT-50S)', ...
    'Location', 'NorthEast');

set(gcf, 'Position', [680, 442, 600, 500]);

set(gca, 'XLim', [0 10], 'YLim', [-0.5 47], 'Box', 'off', 'Tickdir', 'out', ...
    'Fontsize', 14, 'TickLength', [0.005, 0.005]);
xlabel('margin K by which baseline is beaten', 'Fontsize', 14);
ylabel('% queries baseline is beaten by at least K', 'Fontsize', 14);
title('Retrieval curve', 'Fontsize', 16);

export_fig '../../plots/paper/percentage_results.pdf' -transparent;