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
f1 = trace(s'*K*s);
%%===


lambda = FParameters.lambda;
if lambda > 0
    ys = FParameters.ys;
    Xs = FParameters.Xs;
    assert(length(ys)==ns);
    yset = unique(ys);
    f2 = zeros(length(yset),1);
    for c = 1 : length(yset)
        cidx = find(ys==yset(c));
        nc = length(cidx);
        uc = mean(Xs(cidx,:));
        xc = Xs(cidx,:) - repmat(uc,nc,1);
        f2(c) = trace(xc*W*W'*xc');
    end
    f2 = sum(f2)*lambda;
else
    f2 = 0;
end

f = f1 + f2;

