function [resmax,res_mean,res_std,result,PreY]= myNMIACCwithmean_LL(U_temp,Y,numclass)

U = U_temp(:,1:numclass); %no

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
maxIter = 50;

for iter = 1:maxIter
    [indx{iter},center{iter},~,sumD{iter}] = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
    [result(iter,:),map_Y{iter}] = Clustering8Measure(Y,indx{iter});
end

[~,max_index] = max(result(:,1),[],1);
resmax = result(max_index,:);
res_mean = mean(result,1);
res_std = std(result,1);
PreY = map_Y{max_index};
% sumD_res_max = sumD{max_index};
% sumD_res_mean = mean(cat(3,sumD{:}),3);
% 
% center_res_max = center{max_index};
% center_res_mean = mean(cat(3,center{:}),3);
