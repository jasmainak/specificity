clear all;

load('../data/specificity_automated_modified.mat');
load('../data/specificity_scores_all.mat');
load('../data/sentences_all.mat');

fp = fopen('../qualitative/automated_scores.html','w');

specificity = specificity(~isnan(specificity_max));
specificity_max = specificity_max(~isnan(specificity_max));
specificity_av = specificity_max(~isnan(specificity_max));
% corr(specificity, specificity_av, 'type', 'spearman')
corr(specificity, specificity_max, 'type', 'spearman')
corr(specificity, specificity_w, 'type', 'spearman')

break;

all_pairs1 = squeeze(all_pairs1);
all_pairs2 = squeeze(all_pairs2);

scores = reshape(mean(scores,3)',length(all_scores),1);

[~, idx_human] = sort(scores,'descend');
[~, idx_automated] = sort(all_scores, 'descend');

%rank_diff = abs(idx_automated - idx_human);

%score_diff = abs(scores - all_scores);
score_diff = all_scores - scores;

[~, idx_score] = sort(score_diff, 'descend');

fprintf(fp, '<html><body><table>');
fprintf(fp, '<tr><th>Automated scores</th><th>Human scores</th></tr>');
for i=1:length(all_pairs1)    
%for i=1:10
    fprintf(fp, '<tr><td width=600>%s<br/>%s<br/>Score=%0.4f<br/><br/></td>',all_pairs1(idx_automated(i), :), all_pairs2(idx_automated(i), :), all_scores(idx_automated(i)));
    fprintf(fp,'<td width=600><br/>%s<br/>%s<br/>Score=%0.4f<br/><br/></td>',all_pairs1(idx_human(i), :), all_pairs2(idx_human(i), :), scores(idx_human(i)));
    fprintf(fp,'<td width=600><br/>%s<br/>%s<br/>Score diff=%0.4f (Human=%0.4f, Automatic=%0.4f)<br/><br/></td></tr>', all_pairs1(idx_score(i), :), all_pairs2(idx_score(i), :), score_diff(idx_score(i)), scores(idx_score(i)), all_scores(idx_score(i))); 
end

fprintf(fp, '</table></body></html>');

fclose(fp);