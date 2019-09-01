function H = tangent(Y,D)
% TANGENT	produces tangent H from unconstrained differential D.
%		H will be the unique tangent vector such that for any
%		tangent W, real(trace(D'*W)) = ip(Y,W,H).  This is not
%		always an orthogonal projection onto the tangent space.
%
%	H = TANGENT(Y,D)
%	Y is expected to satisfy Y'*Y=I
%	D is expected to be the same size as Y
%	H will satisfy H = tangent(Y,H0)
%
% role	geometric implementation, this function helps produce the gradient and
%	covariant hessian.
	global SGParameters;
	met = SGParameters.metric;
	vert = Y'*D;
	verts = (vert+vert')/2;
	if (met==0)
		H = D - Y*verts;
	elseif (met==1)
		H = D - Y*verts;
	elseif (met==2)
		H = D - Y*vert';
	end
	if (norm(H,'fro')>0 & norm(vert,'fro')/norm(H,'fro')>1e6)
		vert = Y'*H;
		verts = (vert+vert')/2;
		H = H-Y*verts;
	end

