clear all; close all;
load('../../data/specificity_alldatasets.mat');
load('../../data/sentences/clipart_500_img_48_sent.mat', 'clipart_urls', 'clipart_sentences');
y = specificity.clipart.mean;

clipart_urls = clipart_urls(~isnan(y)); clipart_sentences = clipart_sentences(~isnan(y), :);
y = y(~isnan(y)); 

[~, sorted_idx] = sort(y, 'descend');

fp = fopen('../../qualitative/clipart_dataset.html','w');

js_code = 'function openClose(e){document.getElementById(e).style.display="block"==document.getElementById(e).style.display?"none":"block"}document.getElementById&&(document.writeln(''<style type="text/css"><!--''),document.writeln(".texter {display:none} @media print {.texter {display:block;}}"),document.writeln("//--></style>"));';
fprintf(fp, '<script language="JavaScript" type="text/javascript">%s</script>', js_code);

for i=1:length(clipart_urls)
    fprintf(fp, '<img src="%s" width="500"><br/>\n', clipart_urls{sorted_idx(i)});
    fprintf(fp, 'Specificity = %0.2f <br/>\n', y(sorted_idx(i)));
    
    sent = '';
    for j=1:size(clipart_sentences,2)
        sent = strcat(sent, clipart_sentences{sorted_idx(i), j}, '<br/>');
    end
    
    fprintf(fp, '<div onClick="openClose(''a%d'')" style="cursor:hand; cursor:pointer"><u>Sentences</u></div><div id="a%d" class="texter">%s<br /><br /></div>', i, i, sent);
    
end

fclose(fp);