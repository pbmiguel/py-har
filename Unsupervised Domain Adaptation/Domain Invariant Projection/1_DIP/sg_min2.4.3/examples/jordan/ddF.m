function ddf = ddF(Y,H)
% ddF	Computes the second derivative of F, that is,
% 	ddf = d/dx dF(Y+x*H).
%	
%	ddf = DDF(Y,H)
%	Y is expected to satisfy Y'*Y = I
% 	H is expected to be the same size as Y
%	ddf will be the same size as Y
%
% role	objective function, the function is called to apply the 
%	unconstrained hessian of F to a vector.
	global FParameters;
	A = FParameters.A;
	Ab = Block(Y);

	dAb = dBlock(Y,H);
	f = A*Y - Y*Ab;
	df = A*H - H*Ab - Y*dAb;
	ddf = A'*df - df*Ab' - f*dAb';
