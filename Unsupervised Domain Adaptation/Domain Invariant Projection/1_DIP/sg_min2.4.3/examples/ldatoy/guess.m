function Y = guess(p)
% GUESS provides the initial starting guess Y for energy minimization.
%
%       Y = GUESS
%       Y will satisfy Y'*Y = I
%
% role  objective function, produces an initial guess at a minimizer 
%       of F.
        global FParameters;
	A=FParameters.A;
	n = length(A);
	[V,D]=eig(A); D=diag(D);
	[D,I] = sort(D); V = V(:,I);
	Y = V(:,1:p);
