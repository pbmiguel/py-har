function f = F(Y)
% F	Computes the energy associated with the stieffel point Y.
% 	In this case, by the equation f = trace(Y'AY)/2.
%
%	f = F(Y)
%	Y is expected to satisfy Y'*Y=I
%
% role	objective function, the routine called to compute the objective 
%	function.
	global FParameters;
	A = FParameters.A;
	f = sum(sum(real(conj(Y).*(A*Y))))/2;
