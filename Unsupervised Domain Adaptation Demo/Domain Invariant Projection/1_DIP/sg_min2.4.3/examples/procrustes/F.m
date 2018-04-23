function f = F(Y)
% F	Computes the energy associated with the stiefel point Y.
% 	In this case, by the equation f = ||AY-YB||^2/2.
%
%	f = F(Y)
%	Y is expected to satisfy Y'*Y=I
%
% role	objective function, the routine called to compute the objective 
%	function.
	global FParameters;
	A = FParameters.A;
	B = FParameters.B;
	q = A*Y-Y*B;
	f = sum(sum(real(conj(q).*(q))))/2;
