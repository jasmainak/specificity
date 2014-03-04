function disp_corr(r, idx, objectnames, top_x)

% function(R, IDX, OBJECTNAMES, TOP_X)
%     Shows the output of sorted correlation if the correlation R,
%     the sorted order IDX, OBJECTNAMES and the number of results TOP_X
%     is specified.

for i=1:top_x
    fprintf('%f %s\n', r(i), objectnames{idx(i)});
end

end