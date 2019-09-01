function ddf = ddF(Y,H)
% ddF   Computes the second derivative of F, that is,
%       ddf = d/dx dF(Y+x*H).
%
%       ddf = DDF(Y,H)
%       Y is expected to satisfy Y'*Y = I
%       H is expected to be the same size as Y
%       ddf will be the same size as Y
%
% role  objective function, the function is called to apply the 
%       unconstrained hessian of F to a vector.
	global FParameters;
	A = FParameters.A;
	c = FParameters.c;
	ddf =  (A*H);
	ddf = ddf+c*( (sum(Y.^2,2)*ones(1,size(Y,2))).*H );
	ddf = ddf+c*( (sum(2*Y.*H,2)*ones(1,size(Y,2))).*Y );
