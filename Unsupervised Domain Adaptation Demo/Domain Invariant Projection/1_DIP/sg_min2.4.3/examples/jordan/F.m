function f = F(Y)
% F	Computes the objective value associated with the stiefel point Y.
% 	In this case, by the equation f = ||AY-YB(Y)||^2/2.
%
%	f = F(Y)
%	Y is expected to satisfy Y'*Y=I
%
% role	objective function, the routine called to compute the objective 
%	function.
	global FParameters;
	A = FParameters.A;
	Ab = Block(Y);
	f = A*Y - Y*Ab; f = sum(sum(real(conj(f).*f)))/2;
