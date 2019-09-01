function W = dgrad(Y,H)
% DGRAD	Computes the tangent vector, W, which results from applying the
% 	geometrically correct hessian at the stiefel point,Y, to the
% 	tangent vector H.
%
%	W = DGRAD(Y,H)
%	Y is expected to satisfy Y'*Y = I
%	H is expected to satisfy H = tangent(Y,H0)
%	W will satisfy W = tangent(Y,W0)
%
% role	geometrized objective function, the is the routine called to apply
%	the covariant hessian of F to a tangent vector.  The analog is
%	W = A*H, here A is the hessian of F.
	global SGParameters;
	met = SGParameters.metric;
	if (met==0)
		W=tangent(Y,ddF(Y,H));
	else
		df = dF(Y); g = tangent(Y,df);
		W = connection(Y,g,H)+...
			dtangent(Y,df,H)+...
			tangent(Y,ddF(Y,H));
	end
% this line removes the grassmann degrees of freedom from Y
	W = nosym(Y,W);
