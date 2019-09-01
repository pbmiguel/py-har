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
        Ss = FParameters.Ss;

	ddf = 0;
	for k=1:size(Ss,3),
		S = Ss(:,:,k);
		dCom = H*S-S*H;
		ddf = ddf+(dCom*S'-S'*dCom);
	end
