% Create html file for displaying images and their sentence descriptions

clear all;
load('../data/sentences_all.mat');

fp = fopen('../qualitative/memorability_sentences.html','w');
fwrite(fp,'<table>');

for i=1:size(sentences,1)
    fwrite(fp,'<tr>');
    
    fwrite(fp,['<td><img src="http://neuro.hut.fi/~mainak/sampled_images/img_' num2str(i) '.jpg"></td>']);
        
    fwrite(fp,'<td>');
    for j=1:5
        fwrite(fp,[sentences{i,j} '<br/>']);
    end
    fwrite(fp,'</td>');
    
    fwrite(fp,'</tr>');
end
fwrite(fp,'</table>');
fclose(fp);