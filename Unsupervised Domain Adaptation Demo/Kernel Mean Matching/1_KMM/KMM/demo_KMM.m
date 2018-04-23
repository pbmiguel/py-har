clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-weights-3.17\matlab');
addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
% you need to install Mosek through here https://www.mosek.com/downloads/details/3/

% parameter
param.C = 1;
epsilon = 1e-6;
B = 10^10;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');


kparam = struct();
kparam.kernel_type = 'gaussian';
[train_kernel,kernel_param]  = getKernel(train_data.train_features, kparam);
test_kernel = getKernel(test_data.test_features, train_data.train_features, kparam);

% compute weight
fprintf('computing weight....\n');
kmm_beta = calc_kmm(train_kernel, test_kernel', epsilon, B);

para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(kmm_beta, train_data.train_labels, [(1:size(train_kernel, 1))' train_kernel], para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;   
ap  = calc_ap(test_data.test_labels, decs);

