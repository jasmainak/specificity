clear all; close all;

addpath('../../library/export_fig/');
addpath('../aux_functions/');

figure; set(gcf, 'Position', [50, 300, 500, 436]); 

x_pos = [0.06, 0.38, 0.7];
y_pos = 0.12; height = 0.8; width = 0.27;

%legend_names = {'instance\_occurence + co\_occurence + instance\_hand'};

%%%%%%%%%% PASCAL DATASET %%%%%%%%%%
load('../../data/predict_specificity/pascal_decaf.mat');

train_size_pascal = train_size;
boundedline(train_size, mean(r_mse,2), std(r_mse, 0, 2)/sqrt(50), 'b');
h1 = plot(train_size, mean(r_mse, 2), '-o', 'color', 'b', ...
          'Markersize',7,'Markerfacecolor','w', 'Linewidth', 2); hold on;
ylabel('Mean Spearman''s \rho','Fontsize',12);

%%%%%%%%%% MEMORABILITY DATASET %%%%%%%%%%
load('../../data/predict_specificity/memorability_decaf.mat', 'r_mse', 'train_size');

boundedline(train_size, mean(r_mse,2), std(r_mse, 0, 2)/sqrt(50), 'g');
h2 = plot(train_size, mean(r_mse, 2), '-d', 'color', 'g', ...
          'Markersize',7,'Markerfacecolor','w', 'Linewidth', 2); hold on;
xlabel('number of training images','Fontsize',12);
set(gca,'Tickdir','out','Box','off','Fontsize',12, 'YLim', [-0.01 0.3], ...
    'TickLength', [0.005, 0.005], 'Fontsize', 12);

%%%%%%%%%% CLIPART DATASET %%%%%%%%%%
load('../../data/predict_specificity/clipart_objectOccurence-objectcoOccurence-xyz-flip-type.mat');

boundedline(train_size, mean(r_mse,2), std(r_mse, 0, 2)/sqrt(50), 'r');
%subplot(1,3,3, 'Position', [x_pos(3), y_pos, width, height]);
h3 = plot(train_size, mean(r_mse, 2), '-s', 'color', 'r', ...
          'Markersize',7,'Markerfacecolor','w', 'Linewidth', 2); hold on;
%plot(gca, train_size, mean(r_baseline, 2), '--d', 'color', 'g', ...
%     'Markersize',7,'Markerfacecolor','w', 'Linewidth', 2); hold on;

xlabel('number of training images','Fontsize',12);
set(gca,'Tickdir','out','Box','off','Fontsize',12); drawnow;

%%%%%%%%%%% BASELINE %%%%%%%%%

%plot(gca, train_size_pascal, mean(r_baseline, 2), '--d', 'color', 'k', ...
%     'Markersize',7,'Markerfacecolor','w', 'Linewidth', 2); hold on;
legend([h2, h3, h1], 'MEM-5S', 'ABSTRACT-50S', 'PASCAL-50S', 'Location', 'Northwest');
title('Predicted specificity', 'Fontsize', 14);

set(gca, 'Layer', 'top');
set(gcf, 'Position', [286, 219, 538, 517]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

export_fig '../../plots/paper/predict_specificity.pdf' -transparent;