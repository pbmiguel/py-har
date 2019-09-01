function Y = guess(P)
% GUESS	provides the initial starting guess Y for energy minimization.
%	Y will be nxP where FParameters.A is an nxn matrix.  Uses a
%	random matrix.
%
%	Y = GUESS
%	Y will satisfy Y'*Y = I
%
% role	objective function, produces an initial guess at a minimizer 
%	of F.
	global FParameters;
	inA = FParameters.A;
	[n,n] = size(inA);
	[Y,r] = qr(randn(n,P),0);
