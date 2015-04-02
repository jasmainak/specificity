% GRID_SEARCH performs grid search to select C and gamma for SVR
%
% AUTHOR: Mainak Jas
function [bestc, bestg] = grid_search(X, y)

optimalg = 1/size(X,2); % 1/number of features
bestcv = Inf;
for log10C=-1:3
    for g = optimalg/2:optimalg/10:optimalg*1.5
        cv = svmtrain2(y, X, ['-s 3 -v 5 -q -c ' num2str(10^log10C) ' -g ', num2str(g)]);
        if (cv < bestcv),
            bestcv = cv; bestc = 10^log10C; bestg = g;
        end
        fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)', 10^log10C, g, cv, bestc, bestg, bestcv);
    end
end

% Refine grid search
for C=bestc/2:bestc/10:bestc*1.5
    for g = optimalg/2:optimalg/10:optimalg*1.5
        cv = svmtrain2(y, X, ['-s 3 -v 5 -q -c ' num2str(C) ' -g ', num2str(g)]);
        if (cv < bestcv),
            bestcv = cv; bestc = C; bestg = g;
        end
        fprintf('\n(C=%g g=%g rate=%g) (best c=%g, best g=%g, rate=%g)', C, g, cv, bestc, bestg, bestcv);
    end
end

end