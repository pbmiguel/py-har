function Y = guess()
% GUESS	provides the initial starting guess Y for energy minimization.
%	In this case it makes a somewhat random guess.
%
%	Y = GUESS
%	Y will satisfy Y'*Y = I
%
% role	objective function, produces an initial guess at a minimizer 
%	of F.
	global FParameters;
	inA = FParameters.A;
	inB = FParameters.B;
	[n,n] = size(inA);
	[p,p] = size(inB);
	Y = randn(n,p);
	Y = (inA*Y)/inB;
	[Y,r] = qr(Y,0);
