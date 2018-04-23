function df = dF(Y)
% dF	Computes the differential of F, that is,
% 	df satisfies real(trace(H'*df)) = d/dx (F(Y+x*H)).
%
%	df = DF(Y)
%	Y is expected to satisfy Y'*Y = I
%	df is the same size as Y
%
% role	objective function, this is the routine called to compute the
%	differential of F.
	[N,P] = size(Y);
	ep = 1e-6;
	for k=1:P, for j=1:N,
		Yp = Y; Yp(j,k) = Yp(j,k)+ep;
		Ym = Y; Ym(j,k) = Ym(j,k)-ep;
		df(j,k) = (F(Yp)-F(Ym))/ep/2;
	end, end
