function i = ip(Y,H1,H2)
% IP	computes the inner produce of H1,H2 which are tangents at the
%	stiefel point Y.
%
%	i = IP(Y,H1,H2)
%	Y is expected to satisfy Y'*Y=I
%	H1,y are expected to satisfy H1 = tangent(Y,H10), H2 = tangent(Y,H20)
%
% role	geometric implementation, the analog of real(H1'*H2)
	global SGParameters;
	met = SGParameters.metric;
	if (met==0)
% unconstrained metric
		i = sum(sum(real(conj(H1).*H2)));
	elseif (met==1)
% euclidean metric
		i = sum(sum(real(conj(H1).*H2)));
	elseif (met==2)
% canonical metric
		i = sum(sum(real(conj(H1).*H2)))-sum(sum(real(conj(Y'*H1).*(Y'*H2))))/2;
	end
