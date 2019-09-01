function i = ip(Y,H1,H2)
% IP	computes the inner produce of H1,H2 which are tangents at the
%	stiefel point Y.
%
%	i = IP(Y,H1,H2)
%	Y is expected to satisfy Y'*Y=I
%	H1,y are expected to satisfy H1 = tangent(Y,H1), H2 = tangent(Y,H2)
%
% role	geometric implementation, the analog of real(H1'*H2)
	global SGParameters;
	met = SGParameters.metric;

	i = 0;
	if (met==0)
% unconstrained metric
		for yi=1:length(Y)
			i = i+sum(sum(real(conj(H1{yi}).*H2{yi})));
		end
	elseif (met==1)
% euclidean metric
		for yi=1:length(Y)
			i = i+sum(sum(real(conj(H1{yi}).*H2{yi})));
		end
	elseif (met==2)
% canonical metric
		for yi=1:length(Y)
			i = i+sum(sum(real(conj(H1{yi}).*H2{yi})))-sum(sum(real(conj(Y{yi}'*H1{yi}).*(Y{yi}'*H2{yi}))))/2;
		end
	end
