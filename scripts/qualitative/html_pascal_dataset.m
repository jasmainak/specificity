clear all; 
addpath('../io/');

fp = fopen('../../qualitative/pascal_dataset.html','w');

[scores_b, scores_w, s, pascal_sentences, m_sentences, pascal_urls] = load_search_parameters('pascal');

specificity = nanmean(scores_w, 2);
[~, sorted_idx] = sort(specificity,'descend');

% CREATE DIRECTORIES
if ~exist('../../qualitative/pascal_sentences','dir');
    mkdir('../../qualitative/pascal_sentences');
    mkdir('../../qualitative/pascal_sentences/histograms');
    mkdir('../../qualitative/pascal_sentences/sentences');
end

for i=1:length(pascal_urls)
    
    fprintf('\nProcessing image ... [%d/%d]', i, length(pascal_urls));
    
    j = sorted_idx(i); 
    
    % HISTOGRAM
    f = figure('visible', 'off');
    hist(scores_w(j,:), 0:0.05:1); hold on;
    h = findobj(gca,'Type','patch');
    set(h(1),'Facecolor','r','Facealpha',0.5);
    hist(scores_b(j,:), 0:0.05:1);
    plot([0, 1.1], [0.1, 0.1],'k');
    set(gca,'XLim',[0 1.1], 'YLim', [0 300], 'Tickdir','out', 'Box','off');

    saveas(f, sprintf('../../qualitative/pascal_sentences/histograms/hist_%d.png', j));
    close all;
        
    % IMAGE SPECIFIC HTML
    filename = sprintf('../../qualitative/pascal_sentences/sentences/sentences_img%d.html',j);
    gp = fopen(filename, 'w');
    
    fprintf(gp, '<table><tr>');
    fprintf(gp, '<td><img src="%s" height="400"></img></td>\n', pascal_urls{j});
    fprintf(gp, '<td><img src="../histograms/hist_%d.png" height="400"></img></td>\n', j);
    fprintf(gp, '</tr></table>');
    fprintf(gp,'Specificity score = %0.2f<br/><br/>',specificity(j));
    for k=1:size(pascal_sentences,2)
        fprintf(gp,'%s<br/>\n', pascal_sentences{j,k});
    end
    
    % MASTER HTML
    fprintf(fp,'<img src="%s" width="500"></img><br/>\n',pascal_urls{j});
    fprintf(fp,'Specificity score = %0.2f&nbsp;&nbsp;&nbsp;',specificity(j));
    fprintf(fp, ['<a href="#" onclick="MyWindow=window.open(''pascal_sentences/sentences/sentences_img%d.html'', ', ...
                 '''MyWindow'', ''width=1200, height=600, scrollbars=yes''); return false;">', ...
                 '[Sentences and histogram]</a><br/>\n'], j);
    
    fclose(gp);
end

fclose(fp);