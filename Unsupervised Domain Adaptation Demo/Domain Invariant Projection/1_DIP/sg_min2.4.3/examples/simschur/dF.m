function df = dF(Q)
% dF    Computes the differential of F, that is,
%       df satisfies real(trace(H'*df)) = d/dx (F(Q+x*H)).
%
%       df = DF(Q)
%       Q is expected to satisfy Q'*Q = I
%       df is the same size as Q
%
% role  objective function, this is the routine called to compute the
%       differential of F.
	global FParameters;
	A = FParameters.A;
	B = FParameters.B;
	mask = FParameters.Mask;
	qAq = Q'*A*Q;
	qBq = Q'*B*Q;
	qAql = ~mask.*qAq;
	qBql = ~mask.*qBq;
	df = Q*(qAq'*qAql+qAq*qAql'+ qBq'*qBql+qBq*qBql');
