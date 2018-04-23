function parameters(A)
% PARAMETERS	initializes the parameters for an instance of a 
%		minimization problem.
%
%	PARAMETERS(A)
%	A is expected to be a square matrix
%
% role	sets up the global parameters used at all levels of computation.

	global FParameters;
	FParameters = [];
	FParameters.A = A;
