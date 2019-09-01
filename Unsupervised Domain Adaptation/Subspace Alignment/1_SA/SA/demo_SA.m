clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.dim = 200;
param.C   = 1;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');

raw_train_features = train_data.train_features';
raw_test_features = test_data.test_features';

% main algorithm
fprintf('performing subspace alignment...\n');
[train_features, test_features] = subspace_alignment(raw_train_features, raw_test_features, param.dim);

kparam = struct();
kparam.kernel_type = 'gaussian';
[K,kernel_param] = getKernel(train_features', kparam);
test_kernel = getKernel(test_features', train_features', kernel_param);

train_kernel    = [(1:size(K, 1))' K];
para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;      
ap  = calc_ap(test_data.test_labels, decs);

