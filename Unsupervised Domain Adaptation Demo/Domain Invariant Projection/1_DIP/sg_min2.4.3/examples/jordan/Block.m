function [B,eigvs] = Block(Y)
% BLOCK	computes the matrix B appearing in the energy function 
%	F = ||AY-YB(Y)||^2 for jordan block problems.
%
%	[B,eigvs] = Block(Y)
% 	Y is expected to be a stiefel point (Y'*Y= I)
%	B will be an upper triangular matrix in staircase form the 
%		size of Y'*Y
%	eigvs will be the eigenvalues along the diagonal of B.  If 
%	FParameters.type == 'orbit' then eigvs == FParameters.eigs.
%
% role	objective function, auxiliary routine used by F, dF, ddF
	global FParameters;
	A = FParameters.A;
	eigvs = FParameters.eigs;
	blocks = FParameters.blocks;
	bundle = FParameters.bundle;

	[n,k] = size(Y); 
	B = Y'*A*Y;
	[ne,ns] = size(blocks);
	off = 0;
	for i=1:ne,
		if (bundle)
			eigvs(i)= trace(B(off+1:off+sum(blocks(i,:)),off+1:off+sum(blocks(i,:))))/sum(blocks(i,:));
		end
		for j=1:ns,
			if (blocks(i,j)>0)
				B(off+1:k,off+1:off+blocks(i,j))= eigvs(i)*eye(k-off,blocks(i,j));
				off = off + blocks(i,j);
			end
		end
	end
