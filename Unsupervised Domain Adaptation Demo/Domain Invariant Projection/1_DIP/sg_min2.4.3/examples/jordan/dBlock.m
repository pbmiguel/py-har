function B = dBlock(Y,H)
% dBLOCK	This is an auxiliary procedure used by ddF which computes
% 	B = d/dx Block(Y+x*H)
%
%	B = DBLOCK(Y,H)
% 	Y is expected to be a stiefel point (Y'*Y= I)
%	H is expected to satisfy size(H)=size(Y)
%	B will be a staircase matrix the size of Y'*Y
%
% role	objective function, auxiliary function for ddF
	global FParameters;
	A = FParameters.A;
	eigvs = FParameters.eigs;
	blocks = FParameters.blocks;
	bundle = FParameters.bundle;

	[n,k] = size(Y); 
	B =H'*A*Y+Y'*A*H;
	[ne,ns] = size(blocks);
	off = 0;
	for i=1:ne,
		if (bundle)
			deigvs(i)= trace(B(off+1:off+sum(blocks(i,:)),off+1:off+sum(blocks(i,:))))/sum(blocks(i,:));
		else
			deigvs = 0*eigvs;
		end
		for j=1:ns,
			if (blocks(i,j)>0)
				B(off+1:k,off+1:off+blocks(i,j))= deigvs(i)*eye(k-off,blocks(i,j));
				off = off + blocks(i,j);
			end
		end
	end
