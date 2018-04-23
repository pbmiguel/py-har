function parameters(A,c)
% PARAMETERS	initializes the parameters for an instance of a 
%		toy lda problem with laplacian A and self coupling
%		constant c.
%
%	PARAMETERS(A,c)
%	A is expected to be a square symmetric matrix
%
% role	sets up the global parameters used at all levels of computation.

	global FParameters;
	FParameters = [];
	FParameters.A = A;
	FParameters.c = c;
