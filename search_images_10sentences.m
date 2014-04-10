% Image search
clear all; warning off; close all;

load('../data/image_search_10sentences.mat');
load('../data/image_search_results.mat','s','reference_idx');

clearvars -except comb;

load('../data/image_search_10sentences_reference_1stsent.mat');

reference_idx = zeros(1,222);

d = 222; sigma_d = 0.1; sigma_s = sigma_d;
            
reference_idx = reference_idx(1:d);
s = s(1:d, 1:d);

train_idx_w(1, :) = [1; zeros(length(comb)-1,1)];

w = {2:3; 2:4; [2:4,6]; [2:4,6:7]; [2:4,6:8]; [2:4,6:9]; [2:4,6:10]; [1:4,6:10]; [1:10]};

for tr=1:length(w)
    mask = zeros(length(comb),1);
        
    pairs = nchoosek(w{tr,:},2);
    
    for i=1:size(pairs,1)
        mask = (comb(:,1)==pairs(i,1) & comb(:,2)==pairs(i,2)) | (comb(:,2)==pairs(i,1) & comb(:,1)==pairs(i,2)) | mask;
    end
    
    train_idx_w(tr,:)=mask;
end

for tr=1:size(train_idx_w,1)
    
    %% TRAINING PHASE
    mu_s = zeros(1,length(sentences)); mu_d = mu_s;
    for idx=1:length(sentences)
        
        y_s = scores_w(idx,find(train_idx_w(tr,:)==1));
        mu_s(idx) = normfit(y_s);
                
        mu_d(idx) = 0.2;        
        
    end
    
    %% TEST PHASE
    r_s = zeros(d, d); r_d = r_s;

    for i = 1:d
        reference_sentences{i, :} = sentences{i, reference_idx(i) + 1};
    end
    
    s(isnan(s(:))) = -Inf; r_s(isnan(r_s(:))) = -Inf;
    
    for im_idx = 1:d
        
        if rem(im_idx,10)==0
            fprintf('.');
        end
        
        for idx=1:length(reference_sentences)
                        
            p_s = normpdf(s(im_idx, idx), mu_s(idx), sigma_s);
            p_d = normpdf(s(im_idx, idx), mu_d(idx), sigma_d);
            
            r_s(im_idx, idx) = p_s/(p_s + p_d);
            r_d(im_idx, idx) = p_d/(p_s + p_d);
            
        end
        
        %% BASELINE       
        
        [~, idx_b] = sort(s(im_idx, :),'descend');      
               
        %% SPECIFICITY
                
        r_s(im_idx, isnan(r_s(im_idx, :))) = -Inf;
        [~, idx_s] = sort(r_s(im_idx, :),'descend'); 
                
        rank_b(im_idx) = find(idx_b==im_idx);
        rank_s(tr, im_idx) = find(idx_s==im_idx);
        
    end
    
    %% STATS
    fprintf('\n\nMedian rank (Baseline): %0.2f', median(rank_b));
    fprintf('\nMedian rank (Specificity): %0.2f', median(rank_s(tr, :)));
    fprintf('\nMean rank(Baseline): %0.2f', mean(rank_b));
    fprintf('\nMean rank(Specificity): %0.2f', mean(rank_s(tr, :)));
    fprintf('\nCount(specificity rank > baseline rank): %d', sum(rank_b<rank_s(tr,:)));
    fprintf('\nCount(specificity rank < baseline rank): %d', sum(rank_s(tr, :)<rank_b));
    fprintf('\nCount(specificity rank = baseline rank): %d\n', sum(rank_s(tr, :)==rank_b));
    
end

% Effect of data base size

% plot(database_size, database_results);
% hold on; plot(database_size, database_results, 'bo','MarkerFaceColor','w');
% set(gca,'XTick',database_size,'Box','off','TickDir','out');
% xlabel('Database size'); ylabel('(mean(rank-baseline) - mean(rank-specificity))/database-size');
% title('Baseline vs Specificity (Image search)');

colors = repmat(0:0.09:0.09*8,3);
    
% ACCURACY PLOT

for tr=1:size(train_idx_w,1)
    for u=1:d
        baseline_top_k(u) = sum(rank_b<=u)/double(d)*100;
        spec_top_k(u) = sum(rank_s(tr, :)<=u)/double(d)*100;
        %logit_top_k(u) = sum(rank_r<=u)/double(curr_idx)*100;
    end
    
    if tr==1
        plot(1:d, baseline_top_k,'b'); hold on; 
    end
    
    plot(1:d, spec_top_k, 'color', colors(:,tr));
    % plot(1:d, logit_top_k, 'g');
    title('Search Results (Effect of changing number of training sentences)');
    xlabel('k','Fontsize',10); ylabel('Percentage of Queries with rank<=k','Fontsize',10);
    %legend('Baseline','Specificity (2C2)','Specificity (3C2)', 'Specificity(4C2)', 'Specificity (5C2)', 'location','SouthEast');
    set(gca,'XLim',[0 d], 'TickDir', 'out', 'Box','off');
end

% plot(1:4, mean(rank_s(1:4,:),2)); hold on; plot(1:4, mean(rank_s(5:8,:),2),'k');
% plot([1,4], [mean(rank_b), mean(rank_b)],'r--');
% plot(1:4, mean(rank_s(1:4, :),2), 'bo', 'MarkerFaceColor', 'w');
% plot(1:4, mean(rank_s(5:8,:),2),'ko', 'MarkerFaceColor','w');
% set(gca, 'TickDir','out','Box','off','XTick',[1:9],'XTickLabel',{'2C2','3C2','4C2','5C2','6C2','7C2','8C2','9C2','10C2'});
% ylabel('Mean Rank'); legend('Specificity (including reference)','Specificity (excluding reference)','Baseline');
% title('Effect of changing number of training sentences');

plot(1:9, mean(rank_s(1:9,:),2)); hold on; 
plot([1,9], [mean(rank_b), mean(rank_b)],'r--');
plot(1:9, mean(rank_s(1:9, :),2), 'bo', 'MarkerFaceColor', 'w');
plot(8, mean(rank_s(8, :),2), 'go', 'MarkerFaceColor','g');
plot(9, mean(rank_s(9, :),2), 'ro', 'MarkerFaceColor','r');
set(gca, 'TickDir','out','Box','off','XTick',[1:10],'XTickLabel',{'2C2','3C2','4C2','5C2','6C2','7C2','8C2','9C2', '10C2'});
ylabel('Mean Rank'); legend('Specificity','Baseline');
title('Effect of changing number of training sentences');