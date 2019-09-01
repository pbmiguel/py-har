function [M,eigvs] = post(Y)
% POST	Computes the matrix which corresponds to the one nearest to
%	FParameters.A with the jordan structure FParameters.blocks
%	and eigenspace Y.
%
%	[M,eigvs] = post(Y)
%	Y is expected to be a stiefel point (Y'*Y=I)
%	M will be the same size as FParameters.A with jordan structure
%	encoded by FParameters.blocks, eigenvalues of eigvs, and eigenspace
%	of Y.
%	eigvs will be the vector of jordan eigenvalues of M.
	global FParameters;

	A = FParameters.A;
	[B,eigvs] = Block(Y);
	M = A - (A*Y-Y*B)*Y';
