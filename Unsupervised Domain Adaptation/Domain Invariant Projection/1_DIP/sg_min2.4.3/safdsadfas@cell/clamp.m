function [nY,nD] = clamp(Y,D)
% CLAMP		reduces small roundoff errors in Y and D which can
%		accumulate during covariant operations that detroy the
%		orthogonality conditions.
%
%	Usages:
%	nY = CLAMP(Y)
%	nD = CLAMP(Y,D)
%	[nY,nD] = CLAMP(Y,D)
%
%	Y is expected to satisfy Y'*Y=I to order epsilon
%	D is expected to be tangent to Y to order epsilon
%	nY will satisfy Y'*Y = I to epsilon^2
%	nD will be tangent to Y to epsilon^2
%
% role	geometric implementation, this function must be called intermittently
%	on the intermediate results to prevent a catatrophic roundoff.
	for yi = 1:length(Y),
		if (nargin == 1)
			nY{in} = Y{in}*(1.5*eye(size(Y{in},2))-0.5*(Y{in}'*Y{in}));
		end
		if (nargin == 2 & nargout == 1)
			vert = Y{in}'*D{in};
			verts = (vert+vert')/2;
% really should be nD but the argument order has us calling this nY
			nY{yi} = D{yi} - Y{yi}*verts;
		end
		if (nargin == 2 & nargout == 2)
			nY{yi} = Y{yi}*(1.5*eye(size(Y{in},2))-0.5*(Y{in}'*Y{in}));	
			vert = Y{in}'*D{in};
			verts = (vert+vert')/2;
			nD{in} = D{in} - Y{in}*verts;
		end

