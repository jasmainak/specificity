% Create html file for displaying images and their sentence descriptions

clearvars -except sentences;

proj_dir = '/home/mainak/Desktop/specificity';

fp = fopen([proj_dir '/data/sampled_images/index.html'],'w');
fwrite(fp,'<table>');

for i=1:size(sentences,1)
    fwrite(fp,'<tr>');
    
    fwrite(fp,['<td><img src="img_' num2str(i) '.jpg"></td>']);
        
    fwrite(fp,'<td>');
    for j=1:5
        fwrite(fp,[sentences{i,j} '<br/>']);
    end
    fwrite(fp,'</td>');
    
    fwrite(fp,'</tr>');
end
fwrite(fp,'</table>');
fclose(fp);