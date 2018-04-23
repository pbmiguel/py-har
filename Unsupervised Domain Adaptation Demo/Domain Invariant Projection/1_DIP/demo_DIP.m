clear all;clc;

addpath('.\utils');
addpath('.\sg_min2.4.3');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.lambda = 0;
param.dim = 200;
param.C = 1;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');


Xs = train_data.train_features';
Xu = test_data.test_features';
   
PP = calc_pca([Xs; Xu]');
Xs = Xs * PP;
Xu = Xu * PP;

sigma = sqrt(0.5/calc_g(Xs));
sigma = 2*sigma^2;

W = trainDIP_CG(train_data.train_labels, Xs, Xu, sigma, param.lambda, param.dim);

train_feature = Xs * W;
test_feature = Xu * W;
clear W;

kparam = struct();
kparam.kernel_type = 'gaussian';
[K, kernel_param] = getKernel(train_feature', kparam);
test_kernel = getKernel(test_feature', train_feature', kernel_param);

train_kernel    = [(1:size(K, 1))' K];
para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;       
ap  = calc_ap(test_data.test_labels, decs);


