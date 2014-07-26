clear all;

%load('../../data/predict_search/pascal/groundtruth_logistic.mat');
load('../../data/predict_search/pascal/predicted_logistic.mat');

rank_vars = fieldnames(rank_baseline);

test_size = test_size(1:length(rank_vars));

for idx=1:length(test_size)
    eval(['rank_s(idx) = mean(rank_specificity.' rank_vars{idx} ');']);
    eval(['rank_b(idx) = mean(rank_baseline.' rank_vars{idx} ');']);
end

subplot(1,2,1);
plot(test_size, rank_s, 'ro-', 'MarkerFacecolor', 'w'); hold on;
plot(test_size, rank_b, 'bo-', 'MarkerFacecolor', 'w');
ylabel('Mean rank', 'Fontsize', 12);
legend('Predicted specificity', 'Baseline');
xlabel('Number of images ranked', 'Fontsize', 12);
title('Predicted vs baseline (Pascal)', 'Fontsize', 14);

subplot(1,2,2);
plot(test_size, rank_b - rank_s, 'bo-', 'MarkerFacecolor', 'w');
ylabel('Baseline rank - predicted specificity rank', 'Fontsize', 12);
xlabel('Number of images ranked', 'Fontsize', 12);
