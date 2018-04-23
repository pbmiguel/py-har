function df = dF(W)
% dF	Computes the differential of F, that is,
% 	de satisfies real(trace(H'*df)) = d/dx (F(Y+x*H)).
%
%	df = DF(Y)
%	Y is expected to satisfy Y'*Y = I
%	df is the same size as Y
%
% role	objective function, this is the routine called to compute the
%	differential of F.
global FParameters;

sigma  = FParameters.sigma;
X = [FParameters.Xs; FParameters.Xt];
ns = size(FParameters.Xs,1);
nt = size(FParameters.Xt,1);
s = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];

XW = X*W;
D = L2_distance_2(XW', XW');
K = exp(-D./sigma);

K = -2/sigma * K * s * s';

df1 = X' * (diag(sum(K)) - K) * X;

df1 = df1 * 2 * W;


%====
lambda = FParameters.lambda;
if lambda > 0
    ys = FParameters.ys;
    Xs = FParameters.Xs;
    assert(length(ys)==ns);
    yset = unique(ys);
    C = zeros(size(Xs,2));
    for c = 1 : length(yset)
        cidx = find(ys==yset(c));
        nc = length(cidx);
        uc = mean(Xs(cidx,:));
        xc = Xs(cidx,:) - repmat(uc,nc,1);
        C = C + xc'*xc;
    end
    df2 = 2*C*W * lambda;
else
    df2 = zeros(size(W));
end

df = df1 + df2;