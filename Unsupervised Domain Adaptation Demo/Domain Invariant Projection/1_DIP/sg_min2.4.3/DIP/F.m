function f = F(W)
% F	Computes the energy associated with the stieffel point Y.
% 	In this case, by the equation f = trace(Y'AY)/2.
%
%	f = F(Y)
%	Y is expected to satisfy Y'*Y=I
%
% role	objective function, the routine called to compute the objective
%	function.
global FParameters;

sigma = FParameters.sigma;
X = [FParameters.Xs; FParameters.Xt];
ns = size(FParameters.Xs,1);
nt = size(FParameters.Xt,1);
s = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
XW = X*W;
D = L2_distance_2(XW', XW');
K = exp(-D./sigma);
f = trace(s'*K*s);