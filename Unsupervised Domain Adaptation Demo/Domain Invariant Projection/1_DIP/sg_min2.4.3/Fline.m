function f = Fline(t,Y,H)
% FLINE	the objective function, F, along the geodesic passing through Y
%	in the direction H a distance t.
%
%	f= FLINE(t,Y,H)
%	Y is expected to satisfy Y'*Y = I.
%	H is expected to satisfy H = tangent(Y,H0)
%	
% role	high level algorithm, basic window dressing for the fmin function
	f = F(move(Y,H,t));
