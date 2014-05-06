% Image search
clear all; warning off; close all;
addpath(genpath('../../library/boundedline/'));

load('../../data/image_search_50sentences_query.mat');
load('../../data/image_search_50sentences_parameters.mat');

[n_images, n_sentences] = size(sentences); 
mu_d = 0.2; sigma_d = 0.1; sigma_s = sigma_d;

% mycluster = parcluster('local'); delete(mycluster.Jobs);

poolobj = parpool;

for run=1:50
        
    for n_tr=2:n_sentences
        
        fprintf('\nRUN = %d, TRAINING SENTENCES = %d\n', run, n_tr);
        
        if n_tr<=48
            choose_sent = randsample(2:n_sentences, n_tr);
        elseif n_tr==49 % include ref sentence
            choose_sent = randsample(1:n_sentences, n_tr);
        else % include ref + query sentence
            choose_sent = randsample(1:n_sentences, n_tr);
        end
        
        mask = zeros(length(comb), 1);
        pairs = nchoosek(choose_sent, 2);
         
        for i=1:size(pairs,1)
            mask = (comb(:,1)==pairs(i,1) & comb(:,2)==pairs(i,2)) | (comb(:,2)==pairs(i,1) & comb(:,1)==pairs(i,2)) | mask;
        end

        train_idx = find(mask == 1);               
        
        % TRAINING PHASE
        mu_s = zeros(1,length(sentences));
        for idx=1:length(sentences)            
            y_s = scores_w(idx,train_idx);
            mu_s(idx) = normfit(y_s);            
        end
                        
        % TEST PHASE
        r_s = zeros(n_images, n_images); r_d = r_s;
        
        for query_idx = 1:n_images
            
            if rem(query_idx,10)==0
                fprintf('.');
            end
            
            for ref_idx=1:n_images
                
                p_s(query_idx, ref_idx) = normpdf(s(query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                p_d(query_idx, ref_idx) = normpdf(s(query_idx, ref_idx), mu_d, sigma_d);
                
                r_s(query_idx, ref_idx) = p_s(query_idx, ref_idx)/(p_s(query_idx, ref_idx) + p_d(query_idx, ref_idx));
                r_d(query_idx, ref_idx) = p_d(query_idx, ref_idx)/(p_s(query_idx, ref_idx) + p_d(query_idx, ref_idx));                                
               
            end
            
            r_s(isnan(r_s(:))) = -Inf;           
           
            % RANKING:: SPECIFICITY        
            [~, idx_s] = sort(r_s(query_idx, :),'descend');                         
            rank_s(run, n_tr, query_idx) = find(idx_s==query_idx);           
        end
         
    end
    
end

delete(poolobj);

% RANKING:: BASELINE
s(isnan(s(:))) = -Inf; 
for query_idx = 1:n_images    
    [~, idx_b] = sort(s(query_idx, :),'descend');
    rank_b(query_idx) = find(idx_b==query_idx);
end

rank_s_mean = squeeze(mean(rank_s,1));
rank_s_err = std(mean(rank_s,3),0,1);

figure;
x_range = 2:n_sentences;
boundedline(x_range, mean(rank_s_mean(x_range,:),2), rank_s_err(x_range));
h1 = plot(x_range, mean(rank_s_mean(x_range,:),2)); hold on; 
h2 = plot([2,n_sentences], [mean(rank_b), mean(rank_b)],'r--');
plot(x_range, mean(rank_s_mean(x_range, :),2), 'bo', 'MarkerFaceColor', 'w');
plot(49, mean(rank_s_mean(49, :),2), 'go', 'MarkerFaceColor','g');
plot(50, mean(rank_s_mean(50, :),2), 'ro', 'MarkerFaceColor','r');

set(gca, 'TickDir','out','Box','off','XTick',[10:10:50], ...
    'XTickLabel',{'10C2','20C2','30C2','40C2','50C2'}, 'Fontsize',12);

ylabel('Mean Rank','Fontsize',12); xlabel('Number of training sentences','Fontsize',12);
legend([h1, h2], 'Specificity','Baseline','Fontsize',12);
title('Effect of changing number of training sentences','Fontsize',14);