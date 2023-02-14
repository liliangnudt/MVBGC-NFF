function [obj] = callobj(numview,alpha,Wei,X,Ap)
global Z 
term1 = 0;
for p = 1 : numview
    term1 = term1 + alpha(p)^2 * norm(Wei{p} * X{p} - Ap{p} * Z,'fro')^2;
end

obj = term1;

end