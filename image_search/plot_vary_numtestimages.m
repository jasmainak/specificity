function plot_vary_numtestimages(dataset, plot_type)

if strcmpi(plot_type, 'groundtruth')
    load(['../../data/predict_search/' dataset '/groundtruth_logistic.mat']);
elseif strcmpi(plot_type, 'predicted')
    load(['../../data/predict_search/' dataset '/predicted_logistic.mat']);
elseif strcmpi(plot_type, 'predicted_1000fold')
    load(['../../data/predict_search/' dataset '/predicted_logistic_1000fold.mat']);
else
    load('../../data/predict_search/pascal/predicted_logistic_1000fold_python.mat'); % XXX : not sure if this is ok
end

rank_s = zeros(1,length(test_size)); rank_b = zeros(1, length(test_size));
for idx=1:length(test_size)
    rank_s(idx) = mean(rank_specificity{idx});
    rank_b(idx) = mean(rank_baseline{idx});
end

subplot(1,2,1);
plot(test_size, rank_s, 'ro-', 'MarkerFacecolor', 'w'); hold on;
plot(test_size, rank_b, 'bo-', 'MarkerFacecolor', 'w');
ylabel('Mean rank', 'Fontsize', 12);
legend([plot_type ' specificity'], 'Baseline', 'Location', 'SouthEast');
xlabel('Number of images ranked', 'Fontsize', 12);
title([plot_type ' vs baseline (' dataset ')'], 'Fontsize', 14);

subplot(1,2,2);
plot(test_size, rank_b - rank_s, 'bo-', 'MarkerFacecolor', 'w');
ylabel(['Baseline rank - ' plot_type ' specificity rank'], 'Fontsize', 12);
xlabel('Number of images ranked', 'Fontsize', 12);

end
