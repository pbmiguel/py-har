function parameters(A,B)
% PARAMETERS    initializes the parameters for an instance of a 
%               a Schilders problem.
%
%       PARAMETERS(A,B)
%       A,B are expected to be real matrices of the same size.
%
% role  sets up the global parameters used at all levels of computation.
	global FParameters;
	FParameters.A = A;
	FParameters.B = B;
	n = length(A);
	Mask = triu(ones(n));
	if isreal(A) && isreal(B)
	  for l=1:2:n-1,
	    Mask(l+1,l)=1;
	  end
	end
	FParameters.Mask = Mask;
