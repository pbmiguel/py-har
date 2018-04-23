function f = F(Q)
% F     Computes the objective value associated with the stiefel point Q.
%       In this case, by the equation 
%	f = sum(sum(abs(qAql).^2/2+abs(qBql).^2/2));
%	where qAql = masked out part of Q'*A*Q and qBql similarly.
%
%       f = F(Q)
%       Q is expected to satisfy Q'*Q=I
%
% role  objective function, the routine called to compute the objective 
%       function.
	global FParameters;
	A = FParameters.A;
	B = FParameters.B;
	mask = FParameters.Mask;
	qAq = Q'*A*Q;
	qBq = Q'*B*Q;
	qAql = ~mask.*qAq;
	qBql = ~mask.*qBq;
	f = sum(sum(abs(qAql).^2/2+abs(qBql).^2/2));
