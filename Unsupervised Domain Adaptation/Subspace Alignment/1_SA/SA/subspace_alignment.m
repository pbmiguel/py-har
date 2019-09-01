function [Target_Aligned_Source_Data, Target_Projected_Data] = subspace_alignment(Source_Data, Target_Data, Subspace_Dim)

% Input: source_data n*d  target_data n*d
% Output: source_data n*d2 target_data n*d2

% Normalize data
Source_Data = NormalizeData(Source_Data);
Source_Data(isnan(Source_Data)) = 0;
Target_Data = NormalizeData(Target_Data);
Target_Data(isnan(Target_Data)) = 0;

% PCA
Xs = calc_pca(Source_Data');
Xt = calc_pca(Target_Data');

% create subspace
if Subspace_Dim>size(Xs,2)
    Subspace_Dim = size(Xs,2);
end
if Subspace_Dim>size(Xt,2)
    Subspace_Dim = size(Xt,2);
end
Xs = Xs(:,1:Subspace_Dim);
Xt = Xt(:,1:Subspace_Dim);

% Subspace alignment and projections
Target_Aligned_Source_Data = Source_Data*(Xs * Xs'*Xt);
Target_Projected_Data = Target_Data*Xt;

end

function Data = NormalizeData(Data)
    Data = Data ./ repmat(sum(Data,2),1,size(Data,2)); 
    Data(isnan(Data)) = 0;
    Data = zscore(Data);  
end
