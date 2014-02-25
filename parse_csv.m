function [headers, data] = parse_csv(datastruct)
% PARSE_CSV parses comma separated value formatted file.
% 
%   [HEADERS, DATA] = parse_csv(DATASTRUCT) returns two cell arrays
%   HEADERS and DATA using DATASTRUCT returned by IMPORTDATA.
%
%   See also IMPORTDATA.

headers = strrep(regexp(datastruct{1},',','split'),'"','');
data = cell(size(datastruct,1)-1, length(headers));

for i=1:length(datastruct)-1    
    temp = regexp(datastruct{i+1}(2:end-1),'","','split');
    data(i,1:length(temp))=temp; clear temp;
end
    

end