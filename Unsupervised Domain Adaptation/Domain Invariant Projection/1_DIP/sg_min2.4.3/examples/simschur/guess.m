function Y = guess()
% GUESS provides the initial starting guess Y for energy minimization.
%
%       Y = GUESS
%       Y will satisfy Y'*Y = I
%
% role  objective function, produces an initial guess at a minimizer 
%       of F.
	global FParameters;
	A = FParameters.A;
	B = FParameters.B;
% use average
	if isreal(A) && isreal(B)
	  [Y,T] = schur(A+B);
	else
	  [Y,T] = schur(A+B,'complex');
	end
% using random guess
%	Y = randn(size(Y));
% use identity
%	Y = eye(size(A));
