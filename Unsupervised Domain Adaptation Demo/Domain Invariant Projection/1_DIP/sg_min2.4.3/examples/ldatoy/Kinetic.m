function K = Kinetic(n)
	K = 2*eye(n) - diag(ones(1,n-1),-1) - diag(ones(1,n-1),1);
