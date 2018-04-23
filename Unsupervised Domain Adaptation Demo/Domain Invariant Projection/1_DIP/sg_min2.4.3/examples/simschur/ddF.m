function ddf = ddF(Q,dQ)
% ddF   Computes the second derivative of F, that is,
%       ddf = d/dx dF(Q+x*dQ).
%
%       ddf = DDF(Q,dQ)
%       Q is expected to satisfy Q'*Q = I
%       dQ is expected to be the same size as Q
%       ddf will be the same size as Q
%
% role  objective function, the function is called to apply the 
%       unconstrained hessian of F to a vector.
	global FParameters;
	A = FParameters.A;
	B = FParameters.B;
	mask = FParameters.Mask;
	qAq = Q'*A*Q;
	qBq = Q'*B*Q;
	qAql = ~mask.*qAq;
	qBql = ~mask.*qBq;
	dqAq = dQ'*A*Q+Q'*A*dQ;
	dqBq = dQ'*B*Q+Q'*B*dQ;
	dqAql = ~mask.*dqAq;
	dqBql = ~mask.*dqBq;
	ddf = dqAq'*qAql+dqAq*qAql'+ dqBq'*qBql+dqBq*qBql';
	ddf = ddf+qAq'*dqAql+qAq*dqAql'+ qBq'*dqBql+qBq*dqBql';
	ddf = Q*ddf;
	ddf = ddf+dQ*(qAq'*qAql+qAq*qAql'+ qBq'*qBql+qBq*qBql');
