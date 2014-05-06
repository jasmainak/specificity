clear all;
% Image search
clear all; warning off; close all;
addpath(genpath('../../library/boundedline/'));

load('../../data/image_search_50sentences_multiplerefs.mat');
load('../../data/image_search_50sentences_parameters.mat');

[n_images, n_sentences] = size(sentences); 
mu_d = 0.2; sigma_d = 0.1; sigma_s = sigma_d;

expt = 'max'; % {'max', 'mean', 'mean_s', 'product'}

% mycluster = parcluster('local'); delete(mycluster.Jobs);
% poolobj = parpool;

for n_tr=48:48%2:n_sentences
    
    fprintf('\nTRAINING SENTENCES = %d\n', n_tr);
    
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
    
    %train_idx = find(mask == 1);
    train_idx = 9:49;
    
    % TRAINING PHASE
    mu_s = zeros(1,length(sentences));
    for idx=1:length(sentences)
        y_s = scores_w(idx,train_idx);
        mu_s(idx) = normfit(y_s);
    end
    
    % TEST PHASE
    r_s = zeros(length(ref_sentences),n_images, n_images); r_d = r_s;
    cum_r_s = r_s; cum_r_d = r_d;
    
    for query_idx = 1:n_images
        
        if rem(query_idx,10)==0
            fprintf('.');
        end
        
        for ref_idx=1:n_images
            
            for refs=1:length(ref_sentences)                                        
                if strcmpi(expt, 'mean')
                    
                    p_s = normpdf(s(refs, query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                    p_d = normpdf(s(refs, query_idx, ref_idx), mu_d, sigma_d);
                
                    r_s(refs, query_idx, ref_idx) = p_s/(p_s + p_d);
                    r_d(refs, query_idx, ref_idx) = p_d/(p_s + p_d);
                    
                    cum_r_s(refs, query_idx, ref_idx) = mean(r_s(1:refs, query_idx, ref_idx));
                    cum_r_d(refs, query_idx, ref_idx) = mean(r_d(1:refs, query_idx, ref_idx));
                    
                elseif strcmpi(expt, 'max')
                    
                    p_s = normpdf(s(refs, query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                    p_d = normpdf(s(refs, query_idx, ref_idx), mu_d, sigma_d);
                
                    r_s(refs, query_idx, ref_idx) = p_s/(p_s + p_d);
                    r_d(refs, query_idx, ref_idx) = p_d/(p_s + p_d);
                    
                    cum_r_s(refs, query_idx, ref_idx) = max(r_s(1:refs, query_idx, ref_idx));
                    cum_r_d(refs, query_idx, ref_idx) = max(r_d(1:refs, query_idx, ref_idx));
                    
                elseif strcmpi(expt, 'mean_s')
                    
                    p_s = normpdf(mean(s(1:refs, query_idx, ref_idx),1), mu_s(ref_idx), sigma_s);
                    p_d = normpdf(mean(s(1:refs, query_idx, ref_idx),1), mu_d, sigma_d);
                
                    cum_r_s(refs, query_idx, ref_idx) = p_s/(p_s + p_d);
                    cum_r_d(refs, query_idx, ref_idx) = p_d/(p_s + p_d);
                elseif strcmpi(expt, 'product')
                    
                    p_s = normpdf(s(refs, query_idx, ref_idx), mu_s(ref_idx), sigma_s);
                    p_d = normpdf(s(refs, query_idx, ref_idx), mu_d, sigma_d);
                
                    r_s(refs, query_idx, ref_idx) = p_s/(p_s + p_d);
                    r_d(refs, query_idx, ref_idx) = p_d/(p_s + p_d);
                    
                    cum_r_s(refs, query_idx, ref_idx) = prod(r_s(1:refs, query_idx, ref_idx));
                    cum_r_d(refs, query_idx, ref_idx) = prod(r_d(1:refs, query_idx, ref_idx));                    
                    
                end
                    
            end                                   
                   
        end
        
        cum_r_s(isnan(cum_r_s(:))) = -Inf;
        
        % RANKING:: SPECIFICITY
        
        for strength=1:length(ref_sentences)
            [~, idx_s] = sort(cum_r_s(strength, query_idx, :),'descend');        
            rank_s(strength, n_tr, query_idx) = find(idx_s==query_idx);
        end
    end
    
end
  
% RANKING:: BASELINE
s(isnan(s(:))) = -Inf; 
for query_idx = 1:n_images
    for strength=1:length(ref_sentences)
        [~, idx_b] = sort(mean(s(1:strength, query_idx, :),1),'descend');
        rank_b(strength, query_idx) = find(idx_b==query_idx);
    end
end

h1 = plot(mean(rank_s(:, 48, :),3)); hold on; plot(mean(rank_s(:, 48, :),3), 'bo', 'Markerfacecolor','w');
h2 = plot(mean(rank_b,2),'r'); plot(mean(rank_b,2),'ro','Markerfacecolor','w');
legend([h1, h2], 'Specificity', 'Baseline');
title('Effect of number of reference sentences','Fontsize',12);
xlabel('Number of reference sentences'); ylabel('Mean Rank');
set(gca,'Box','off','Tickdir','out');

% delete(poolobj);