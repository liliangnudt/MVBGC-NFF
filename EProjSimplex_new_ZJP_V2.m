function [x ft] = EProjSimplex_new_ZJP_V2(H_i,F_i, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%

if nargin < 3
    k = 1;
end

ft=1;
n = length(F_i);

% H_i = H_i;
% v0 = v-mean(v) + k/n;
v0 = -F_i./H_i;
% if min(v0)~=0
%     min(v0)
% end
%vmax = max(v0);
vmin = min(v0);
if 1 % vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m./H_i;
        posidx = v1>0;
        npos = sum(posidx);
%         g = -npos;
        f = sum(v1(posidx)) - k;
        d=sum(1./H_i(posidx));
        if d==0
            lambda_m = lambda_m - 100;
        else
            lambda_m = lambda_m + f/sum(1./H_i(posidx));
        end
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break
        end
    end
    x = max(v1,0);

else
    x = v0;
end