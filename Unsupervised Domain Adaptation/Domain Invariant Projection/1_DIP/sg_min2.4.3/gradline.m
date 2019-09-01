function mg = gradline(t,Y,H)
% GRADLINE the square magnitude of the gradient of the objective function,
%	F, along the geodesic passing through Y in the direction H a
%	distance t.
%
%	f= FLINE(t,Y,H)
%	Y is expected to satisfy Y'*Y = I.
%	H is expected to satisfy H = tangent(Y,H0)
%	
% role	high level algorithm, basic window dressing for the fmin function
	Y = move(Y,H,t);
	g = grad(Y);
	mg = ip(Y,g,g); 