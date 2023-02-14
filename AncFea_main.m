function [UU,Ap,Z_final,Wei_final,alpha,obj] = AncFea_main(X,Y,Anchor)

global Z alpha_sum

%% initialize
rng(0);
maxIter = 50 ;

numsample = size(Y,1);
numview = length(X);
numclass = length(unique(Y));
m = Anchor;

for p = 1 : numview
    X_dim(p) = size(X{p},1);
end
%% initialize alpha
alpha = ones(1,numview)/numview;
%% initialize Wei
Wei = cell(numview,1);
for p = 1:numview
    Wei{p} = diag(1/X_dim(p)*ones(1,X_dim(p)));
end
%% initialize A
for p = 1 : numview
    Ap{p} = zeros(X_dim(p),m);
end
%% initialize Z
Z = 0;
XX = [];
for p = 1 : numview
    XX = [XX;X{p}];
end
[XU,~,~]=svds(XX',m);

[IDX,~] = kmeans(XU,m, 'MaxIter',200,'Replicates',30);
for i = 1:numsample
    Z(IDX(i),i) = 1;
end
Z = Z/(m) + (m-1)/m/m;

%%
flag = 1;
iter = 0;
obj = [];
%%
while flag
    iter = iter + 1;
    
    %% optimize Ap
    for p = 1:numview
        AB = Wei{p} * X{p} * Z';
        [UAB,~,VAB] = svd(AB,'econ');
        Ap{p} = UAB*VAB';
    end
       
    %% optimize Z
    ftemp = 0;
    alpha_sum = 0;
    for p = 1 : numview
        alpha2 = alpha(p)^2;
        alpha_sum = alpha_sum + alpha2;
        ftemp = ftemp - 2 * alpha2 * Ap{p}' * Wei{p} * X{p};
    end
    
    for j = 1:numsample
        Z_hat = -ftemp(:,j)/2/(alpha_sum);
        [Z(:,j)] = EProjSimplex_new(Z_hat);
    end
        
    %% optimze Wei

        WeiG = cell(numview,1);
        Weif = cell(numview,1);
        Wei_temp = cell(numview,1);
        for p = 1 : numview
            WeiG{p} = 2*diag(diag(X{p} * X{p}'));
            Weif{p} = -2*diag(X{p} * Z' * Ap{p}');
            Wei_temp{p} = EProjSimplex_new_ZJP_V2(diag(WeiG{p}),Weif{p});
            Wei{p} = diag(Wei_temp{p});
        end
            
    %% optimize alpha
    M = zeros(numview,1);
    for p = 1 : numview
        M(p) = norm(Wei{p} * X{p} - Ap{p} * Z,'fro')^2;
    end
    Mfra = M.^(-1);
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;
    
    %% Obj
    [obj(end+1)] = callobj(numview,alpha,Wei,X,Ap);
    
    %%
    Z_final = Z;
    
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,~]=svd(Z_final','econ');
        flag = 0;
    end
end
Wei_final = Wei_temp;
end