function df = dF(Y)
% dF    Computes the differential of F, that is,
%       df satisfies real(trace(H'*df)) = d/dx (F(Y+x*H)).
%
%       df = DF(Y)
%       Y is expected to satisfy Y'*Y = I
%       df is the same size as Y
%
% role  objective function, this is the routine called to compute the
%       differential of F.
	global FParameters;
	A = FParameters.A;
	c = FParameters.c;
	df =  (A*Y)+c*( (sum(Y.^2,2)*ones(1,size(Y,2))).*Y );
