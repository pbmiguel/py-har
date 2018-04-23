function Y = guess()
% GUESS	provides the initial starting guess Y for energy minimization.
%
%	Y = GUESS
%	Y will satisfy Y'*Y = I
%
% role	objective function, produces an initial guess at a minimizer 
%	of F.
	global FParameters;
	inA = FParameters.A;
	blocks = FParameters.blocks;
	eigs = FParameters.eigs;

	[n,k] = size(inA);
	[ne,ns] = size(blocks);
	off = 0;
	Q = eye(n); A = inA;
	for i=1:ne, for j=1:ns,
		if (blocks(i,j) ~= 0)
			[u,s,v] = svd(A(off+1:n,off+1:n)-eye(n-off)*eigs(i));
			v = v(:,n-off:-1:1);
			q = [eye(off) zeros(off,n-off); zeros(n-off,off) v];
			A = q'*A*q;
			Q = Q*q;
			s = diag(s); s = flipud(s); 
			off = off + blocks(i,j);
		end
	end, end
	Y = Q(:,1:off);
