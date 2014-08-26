clear all;

addpath('utils/');
addpath('../aux_functions/');

load('search_parameters_pascal.mat');
load('../../data/predict_search/pascal/predicted_specificity_1000fold.mat');
load('../../data/predict_search/pascal/groundtruth_specificity.mat');

rank_b = baseline_search(s);
rank_s = specificity_search(s, y_pred, z_pred);

bins = 50;
bin_width = 1000/bins;
bin_centers = bin_width/2: bin_width: 1000 - bin_width/2;

rankb_gt_ranks = zeros(size(rank_b));
ranks_gt_rankb = zeros(size(rank_b));
ranks_eq_rankb = zeros(size(rank_b));
for r=1:length(rank_b)
    idx = find(rank_b == r);
    rankb_gt_ranks(r) = sum(rank_s(idx) < rank_b(idx));
    ranks_gt_rankb(r) = sum(rank_b(idx) < rank_s(idx));
    ranks_eq_rankb(r) = sum(rank_b(idx) == rank_s(idx));
end

y1 = hist(rank_b, bins);
y2 = hist(rank_s, bins);
bar(bin_centers, [y1' y2'], 1);
title('Histogram of ranks', 'Fontsize', 14);
legend('Baseline', 'Specificity');
set(gca, 'box' ,'off', 'tickdir', 'out', 'TickLength', [0.001 0.001]);

for i=1:length(bin_centers)
    bin_l = bin_centers(i) - bin_width/2 + 1;
    bin_h = bin_centers(i) + bin_width/2;
    rankb_gt_ranks_binned(i) = sum(rankb_gt_ranks(bin_l : bin_h));
    ranks_gt_rankb_binned(i) = sum(ranks_gt_rankb(bin_l : bin_h));
    ranks_eq_rankb_binned(i) = sum(ranks_eq_rankb(bin_l : bin_h));
end

figure;
bar(bin_centers, [ranks_gt_rankb_binned' rankb_gt_ranks_binned' ranks_eq_rankb_binned'], 1);
title('Histogram of counts (bin size=20)', 'Fontsize', 14);
xlabel('Baseline rank', 'Fontsize', 12); ylabel('count', 'Fontsize',12);
legend('Baseline < Specificity', 'Specificity < Baseline', 'Specificity = Baseline');
set(gca, 'box' ,'off', 'tickdir', 'out', 'TickLength', [0.001 0.001]);
