function f= F(Y)
% F(Y)	computes the energy of a toy lda particle in a box configuration.
%	F(Y) = 1/2*trace(Y'*A*Y)+c*1/4*sum(sum(Y.^2,2).^2,1).
%
%       f = F(Y)
%       Y is expected to satisfy Y'*Y=I
%
% role  objective function, the routine called to compute the objective 
%       function.
	global FParameters;
	A = FParameters.A;
	c = FParameters.c;
	f =  trace(Y'*(A*Y))/2+c*sum(sum(Y.^2,2).^2,1)/4;
