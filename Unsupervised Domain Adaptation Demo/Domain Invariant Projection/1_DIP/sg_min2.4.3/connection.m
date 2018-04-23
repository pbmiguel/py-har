function C = connection(Y,H1,H2)
% CONNECTION	Produces the christoffel symbol, C, for either the canonical 
%	or euclidean connections at the point Y.  
%
%	C = CONNECTION(Y,H1,H2) 
%	Y is expected to satisfy Y'*Y = I
%       H1 and H2 satisfy H1=tangent(Y,H10) and H2=...(similarly)
%	C will be a matrix of the same size as Y
%	
% role	geometric implementation, an important term in computing the hessian
%	and in defining the geodesics.
	global SGParameters;
	met = SGParameters.metric;
	if (met==0)
% the unconstrained connection
		C = zeros(size(Y));
	elseif (met==1)
% the euclidean connection for stiefel
		C = Y*(H1'*H2+H2'*H1)/2;
	elseif (met==2)
% the canonical connection for the stiefel
		b = H1'*H2-H1'*Y*Y'*H2;
		C = (H1*H2'*Y+H2*H1'*Y)/2+Y*(b+b')/2;
	end
