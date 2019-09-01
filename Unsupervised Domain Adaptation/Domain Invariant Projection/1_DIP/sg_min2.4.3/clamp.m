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
%	on the intermediate results to prevent a catatrophic roundoff error.
%	It is called in every possible occurrence of roundoff in this
%	Template.  In practise, one should only call it as needed.
%
% note  this is equivalent to multiplication by an approximation of
%	the matrix inverse square root of Y'*Y
clamp_off=0;
if (clamp_off)
    if (nargin==1)
        nY = Y;
    end
    if (nargin==2 & nargout==1)
        nY = D;
    end
    if (nargin==2 & nargout==2)
        nY = Y;
        nD = D;
    end
    return
end
if (nargin == 1)
    nY = Y*(1.5*eye(size(Y,2))-0.5*(Y'*Y));
end
if (nargin == 2 & nargout == 1)
    vert = Y'*D;
    verts = (vert+vert')/2;
    % really should be nD but the argument order has us calling this nY
    nY = D - Y*verts;
end
if (nargin == 2 & nargout == 2)
    nY = Y*(1.5*eye(size(Y,2))-0.5*(Y'*Y));
    vert = Y'*D;
    verts = (vert+vert')/2;
    nD = D - Y*verts;
end
