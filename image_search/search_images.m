% Image search
clear all; warning off;

database_size = [25, 50, 100, 200, 300, 400, 500, 600, 700]; % Try
%different database sizes

database_results = zeros(1,length(database_size));
d_idx = 1; d=888;

close all; %clearvars -except database_results database_size d d_idx pr priors rank_po;
load('../../data/image_search_results.mat');
load('../../data/sentences_all.mat');
load('../../data/image_search_parameters.mat', 'scores_w', 'scores_b');

curr_idx = d;

r_d = r_d(1:d, 1:d); r_s = r_s(1:d, 1:d); reference_idx = reference_idx(1:d);
s = s(1:d, 1:d); scores_b = scores_b(1:d, :, :); scores_w = scores_w(1:d, :, :);
sentences = sentences(1:d, :);

g = fopen('../../qualitative/search_output.html','w');
fprintf(g,'<html><body>First row=BASELINE<br/>Second row=SPECIFICITY');
fprintf(g,'<br/><br/>&mu;<sub>s</sub> and &mu;<sub>d</sub> parameter values shown as image tooltips. Target image in red border.');

train_idx_b = 1:4;
%train_idx_w = [1; 1, 2, 5; 1, 2, 3, 5, 6, 8; 1:10]; % [1:10];
train_idx_w = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
               1, 1, 0, 0, 1, 0, 0, 0, 0, 0; ...
               1, 1, 1, 0, 1, 1, 0, 1, 0, 0; ...
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
     
sigmad = 0.05:0.05:0.4; % Try different sigmas
%priors = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]; % Try different priors

for tr=4%1:size(train_idx_w,1)

%% TRAINING PHASE
mu_s = zeros(1,length(sentences)); mu_d = mu_s;
for idx=1:length(sentences)
    y_s = scores_w(idx,find(train_idx_w(tr,:)==1));
    mu_s(idx) = normfit(y_s);
    
    y_d = scores_b(idx, train_idx_b, :);
    mu_d(idx) = normfit(y_d(:));
    
    X = cat(1, y_s', y_d(:));
    labels = cat(1, ones(length(y_s),1), zeros(length(y_d(:)),1));
    
    % shuffle and fit logistic
    % randomorder = randperm(length(X)); X = X(randomorder); labels = labels(randomorder);
    % B(idx, :) = glmfit(X, labels, 'binomial', 'logit');
end

%% TEST PHASE
r_s = zeros(curr_idx, length(sentences)); r_d = r_s;
for k=2:2%1:length(sigmad)
    
    f = fopen(sprintf('../../qualitative/search_diagnostics_%d.txt', k), 'w');
    
    fprintf(f, '\n\nSIGMA = %0.2f\n',sigmad(k));
    fprintf('\n\nSIGMA = %0.2f\n',sigmad(k));
    for i = 1:d
        reference_sentences{i, :} = sentences{i, reference_idx(i) + 1};
    end
    
    s(isnan(s(:))) = -Inf; r_s(isnan(r_s(:))) = -Inf;
    
    for im_idx = 1:curr_idx
        
        if rem(im_idx,10)==0
            fprintf('.');
        end
        
        for idx=1:length(reference_sentences)
            
            sigma_d = sigmad(k);
            sigma_s = sigma_d;
            
            p_s = normpdf(s(im_idx, idx), mu_s(idx), sigma_s);
            p_d = normpdf(s(im_idx, idx), mu_d(idx), sigma_d);
            
            r_s(im_idx, idx) = p_s/(p_s + p_d);
            r_d(im_idx, idx) = p_d/(p_s + p_d);
            
            % r_logistic(im_idx, idx) = glmval(squeeze(B(idx, :))', s(im_idx, idx), 'logit');
        end
        
        %% BASELINE
        fprintf(f, '\n[%d] Query sentence : %s\n\n', im_idx, sentences{im_idx, 5});
        fprintf(g, '\n<br/><br/>[%d] Query sentence : %s<br/>', im_idx, sentences{im_idx, 5});
        
        [~, idx_b] = sort(s(im_idx, :),'descend');
        
        fprintf(g, '<br/>');
        for i=1:8
            fprintf(f, '%s (s=%0.2f)\n', reference_sentences{idx_b(i)}, s(im_idx, idx_b(i)));
            
            if idx_b(i)~=im_idx
                color='style="border:2px solid white"';
            else
                color='style="border:2px solid red"';
            end
            fprintf(g, '\n\t<img src="http://neuro.hut.fi/~mainak/sampled_images/img_%d.jpg" width="128" %s title="mu_s=%0.2f, mu_d=%0.2f"></img>', idx_b(i), color, mu_s(idx_b(i)), mu_d(idx_b(i)));
        end
        
        %% SPECIFICITY
        
        fprintf(f, '\nSPECIFICITY\n');
        
        r_s(im_idx, isnan(r_s(im_idx, :))) = -Inf;
        [~, idx_s] = sort(r_s(im_idx, :),'descend');
        
        fprintf(g, '<br/>');
        for i=1:8
            fprintf(f, '%s (s=%0.2f, r_s=%0.2f)\n', reference_sentences{idx_s(i)}, s(im_idx, idx_s(i)), r_s(im_idx, idx_s(i)));
            
            if idx_s(i)~=im_idx
                color='style="border:2px solid white"';
            else
                color='style="border:2px solid red"';
            end
            fprintf(g, '\n\t<img src="http://neuro.hut.fi/~mainak/sampled_images/img_%d.jpg" width="128" %s title="mu_s=%0.2f, mu_d=%0.2f"></img>', idx_s(i), color, mu_s(idx_s(i)), mu_d(idx_s(i)));
        end
        
        %% LOGISTIC REGRESSION
        
        %             fprintf(f, '\nLOGISTIC REGRESSION\n');
        %             r_logistic(im_idx, isnan(r_logistic(im_idx, :))) = -Inf;
        %             [~, idx_r] = sort(r_logistic(im_idx, :), 'descend');
        
        %             fprintf(g, '<br/>');
        %             for i=1:8
        %
        %                 if idx_r(i)~=im_idx
        %                     color='style="border:2px solid white"';
        %                 else
        %                     color='style="border:2px solid red"';
        %                 end
        %                 fprintf(g, '\n\t<img src="http://neuro.hut.fi/~mainak/sampled_images/img_%d.jpg" width="128" %s title="mu_s=%0.2f, mu_d=%0.2f"></img>', idx_r(i), color, mu_s(idx_r(i)), mu_d(idx_r(i)));
        %             end
        
        rank_b(im_idx) = find(idx_b==im_idx);
        rank_s(tr, im_idx) = find(idx_s==im_idx);
        % rank_r(im_idx) = find(idx_r==im_idx); % Logistic regression
        
    end
    
%% STATS
fprintf('\nMedian rank (Baseline): %0.2f', median(rank_b));
fprintf('\nMedian rank (Specificity): %0.2f', median(rank_s(tr, :)));
fprintf('\nMean rank(Baseline): %0.2f', mean(rank_b));
fprintf('\nMean rank(Specificity): %0.2f', mean(rank_s(tr, :)));
fprintf('\nCount(specificity rank > baseline rank): %d', sum(rank_b<rank_s(tr,:)));
fprintf('\nCount(specificity rank < baseline rank): %d', sum(rank_s(tr, :)<rank_b));
fprintf('\nCount(specificity rank = baseline rank): %d', sum(rank_s(tr, :)==rank_b));

%fclose(f);

end

database_results(d_idx) = (mean(rank_b) - mean(rank_s(tr, :)))/d; d_idx = d_idx + 1;

fprintf(g,'</html></body>');
fclose(g);

end

% Effect of data base size

% plot(database_size, database_results);
% hold on; plot(database_size, database_results, 'bo','MarkerFaceColor','w');
% set(gca,'XTick',database_size,'Box','off','TickDir','out');
% xlabel('Database size'); ylabel('(mean(rank-baseline) - mean(rank-specificity))/database-size');
% title('Baseline vs Specificity (Image search)');

colors = {'r','c','g','k'};
    
% ACCURACY PLOT

for tr=1:size(train_idx_w,1)
    for u=1:d
        baseline_top_k(u) = sum(rank_b<=u)/double(curr_idx)*100;
        spec_top_k(u) = sum(rank_s(tr, :)<=u)/double(curr_idx)*100;
        %logit_top_k(u) = sum(rank_r<=u)/double(curr_idx)*100;
    end
    
    if tr==1
        plot(1:d, baseline_top_k,'b'); hold on; 
    end
    
    plot(1:d, spec_top_k, colors{tr});
    % plot(1:d, logit_top_k, 'g');
    title('Search Results (Effect of changing number of training sentences)');
    xlabel('k','Fontsize',10); ylabel('Percentage of Queries with rank<=k','Fontsize',10);
    legend('Baseline','Specificity (2C2)','Specificity (3C2)', 'Specificity(4C2)', 'Specificity (5C2)', 'location','SouthEast');
    set(gca,'XLim',[0 d], 'TickDir', 'out', 'Box','off');
end

plot(1:4, mean(rank_s,2)); hold on; plot([1,4], [mean(rank_b), mean(rank_b)],'r--');
plot(1:size(train_idx_w,1), mean(rank_s,2), 'bo', 'MarkerFaceColor', 'w');
set(gca, 'TickDir','out','Box','off','XTick',[1:4],'XTickLabel',{'2C2','3C2','4C2','5C2'});
ylabel('Mean Rank'); legend('Specificity','Baseline');
title('Effect of changing number of training sentences');