function [A,B] = noisy()
	[A,B] = exact;
	randn('state',0);
	A = A + 1e-4*randn(size(A));
	B = B + 1e-4*randn(size(B));
