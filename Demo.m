clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
Dataset_Name = {'MSRCV1'};
Dataset_Path = 'C:\2021-PRMI\MultiView Dataset\';
resPath = './Res/';
addpath(resPath);

metric = {'ACC','NMI','Purity','Fscore','Precision','Recall','ARI','Entropy'};

for Data_index = [1]
    % load data & make folder
    dataName = Dataset_Name{Data_index}; disp(dataName);
    load(strcat(Dataset_Path,dataName));
    
    matpath = strcat(resPath,dataName); %保存结果
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    %%
    numsample = size(Y,1);
    numview = length(X); %原始输入的是 n * dp
    numclass = length(unique(Y));
    
    for p = 1:numview
        X{p} = mapstd(X{p}',0,1);
        X_dim(p) = size(X{p},1);
        X_dim_min = min(X_dim);
        X_dim_max = max(X_dim);
    end
    
    for p = 1:numview
        index = sum(abs(X{p}),2) > 1e-8;
        X{p} = X{p}(index,:);
        X_dim(p) = sum(index);
    end
    
    Anchor = [1]*numclass;
    %%
    for Anc_index = 1:length(Anchor)
        
        if Anchor(Anc_index) > numsample | Anchor(Anc_index) > X_dim_min
            continue
        end
        tic;
        [U,Ap,Z,Wei,alpha,obj] = AncFea_main(X,Y,Anchor(Anc_index));
        [res_max,res_mean,res_std,result,PreY]= myNMIACCwithmean_LL(U,Y,numclass);
        Time_all(Anc_index)  = toc;
        
        fprintf('Anchor:%4.0f \t Time:%4.2f \t ACC:%4.2f \t NMI:%4.2f \t Pur:%4.2f \t Fscore:%4.2f \n',...
            [Anchor(Anc_index) Time_all(Anc_index) res_max(1)*100 res_max(2)*100 res_max(3)*100 res_max(4)*100]);
        fprintf('Anchor:%4.0f \t Time:%4.2f \t ACC:%4.2f \t NMI:%4.2f \t Pur:%4.2f \t Fscore:%4.2f \n',...
            [Anchor(Anc_index) Time_all(Anc_index) res_mean(1)*100 res_mean(2)*100 res_mean(3)*100 res_mean(4)*100]);
    end
end
