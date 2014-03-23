clear all; close all;

load('../data/memorability_sorted.mat');
load('../data/memorability_mapping.mat','mapping');
load('../data/specificity_scores_all.mat');
load('../data/sentences_all.mat');

scores = scores*9 + 1;
data_collected = 888;

[specificity, idx] = sort(specificity,'descend');

fprintf('\nWriting html ... ');

fp = fopen('../data/sampled_images/index_specificity.html','w');

fwrite(fp,'<table>');
for i=1:length(specificity)
        
    % Show images and the memorability and specificity scores
    fwrite(fp,'<tr>');    
    fwrite(fp,['<td><img src="img_' num2str(idx(i)) '.jpg"><br/>']);
    fwrite(fp,sprintf('Specificity score=%0.2f<br/>',specificity(i)));       
    fwrite(fp,sprintf('Memorability score=%0.2f</td>',mem(mapping(idx(i))) )); 
    
    % Write out sentences
    fwrite(fp,'<td width="500">');
    for j=1:5
        fwrite(fp,[num2str(j) '. ' sentences{idx(i),j} '<br/>']);
    end
    fwrite(fp,'</td>');
    
    % Show similarity ratings
    fwrite(fp,'<td><table border="1"><tr>');
    
    fwrite(fp,'<th>  </th><th width=50> 1 </th><th> 2 </th><th> 3 </th><th> 4 </th><th> 5 </th>');
    fprintf(fp,'</tr><tr><th width=50> 1 </th>');
    
    fwrite(fp,sprintf('<td></td><td>%d, %d, %d</td>',scores(idx(i),1,1), scores(idx(i),1,2), scores(idx(i),1,3)));
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),2,1), scores(idx(i),2,2), scores(idx(i),2,3))); 
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),3,1), scores(idx(i),3,2), scores(idx(i),3,3)));
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),4,1), scores(idx(i),4,2), scores(idx(i),4,3)));
    
    fprintf(fp,'</tr><tr><th> 2 </th>');
    
    fwrite(fp,sprintf('<td></td><td></td><td>%d, %d, %d</td>',scores(idx(i),5,1), scores(idx(i),5,2), scores(idx(i),5,3)));
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),6,1), scores(idx(i),6,2), scores(idx(i),6,3))); 
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),7,1), scores(idx(i),7,2), scores(idx(i),7,3)));
    
    fprintf(fp,'</tr><tr><th> 3 </th>');
    
    fwrite(fp,sprintf('<td></td><td></td><td></td><td>%d, %d, %d</td>',scores(idx(i),8,1), scores(idx(i),8,2), scores(idx(i),8,3)));
    fwrite(fp,sprintf('<td>%d, %d, %d</td>',scores(idx(i),9,1), scores(idx(i),9,2), scores(idx(i),9,3))); 
    
    fprintf(fp,'</tr><tr><th> 4 </th>');
    
    fwrite(fp,sprintf('<td></td><td></td><td></td><td></td><td>%d, %d, %d</td>',scores(idx(i),10,1), scores(idx(i),10,2), scores(idx(i),10,3)));
    
    fwrite(fp,'</tr></table></td>');
    
    fwrite(fp,'</tr>');
    
end
fwrite(fp,'</table>');

fclose(fp);
fprintf('[Done]');