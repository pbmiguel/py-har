function G = grad(Y)
% GRAD	computes the gradient of the energy F at the point Y.
%
%	G = GRAD(Y)
%	Y is expected to satisfy Y'*Y=I.
%	G will satisfy G = tangent(Y,G0)
%
% role	geometrized objective function, this is the routine called to produce
%	the geometric gradient of the unconstrained differential of F.
%	The analog is G = A*Y-b.
G = tangent(Y,dF(Y));
