function dim = dimension(Y0)
% DIMENSION	correctly counts the dimension, dim (stiefel dimension minus 
%	the grassmann degrees of freedom).
%  
%	dim = DIMENSION
%
% role	geometrized objective function, used to set the parameters describing
%	the particular submanifold of the stiefel manifold on which the
%	computation will occur.
	global SGParameters;
	cm = SGParameters.complex;
	part = SGParameters.partition;

	dim = 0;
	for yi=1:length(Y0)
		[N,P] = size(Y0{yi});
		np = length(part{yi});
		if (~cm)
			dim = dim+N*P;
			dim = dim - P*(P+1)/2;
			for i=1:np,
				k = length(part{yi}{i});
				dim = dim - k*(k-1)/2;
			end 
		else
			dim = dim+2*N*P;
			dim = dim - P^2;
			for i=1:np,
				k = length(part{yi}{i});
				dim = dim - k^2;
			end
		end
	end



