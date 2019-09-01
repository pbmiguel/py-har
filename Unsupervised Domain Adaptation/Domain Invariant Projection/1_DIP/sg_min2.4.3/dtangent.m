function T = dtangent(Y,H,dY)
% DTANGENT 	computes the differential of the tangent map.  That is
% 	T = d/dt tangent(params,Y+t*dY,H).
%
%	T = DTANGENT(Y,H,dY)
%	Y is expected to satisfy Y'*Y = I
%	H is expected to be the same size as Y
%	dY is expected to be the same size as Y
%	T will be the same size as Y
%
% role	geometric implementation, helps to produce a geometrically 
%	correct covariant hessian. 
	global SGParameters;
	met = SGParameters.metric;
	vert = Y'*H; verts = (vert+vert')/2;
	dvert = dY'*H; dverts = (dvert+dvert')/2;
	if (met==0)
		T = -dY*verts-Y*dverts;
	elseif (met==1)
		T = -dY*verts-Y*dverts;
	elseif (met==2)
		T = -dY*vert'-Y*dvert';
	end
