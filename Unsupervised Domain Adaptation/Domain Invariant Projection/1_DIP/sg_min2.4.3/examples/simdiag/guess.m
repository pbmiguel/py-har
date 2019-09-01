function Y0 = guess()
% GUESS provides the initial starting guess Y for an F minimization.
%
%       Y = GUESS
%       Y will satisfy Y'*Y = I
%
% role  objective function, produces an initial guess at a minimizer 
%       of F.
	global FParameters;
	Ss = FParameters.Ss;
	Q = 0;
	
% take an average
%	av = Ss(:,:,1);
%	for k=2:size(Ss,3)
%		av = av +Ss(:,:,k);
%	end
%	[Y0,D] = eig(av);
% using random guess
%	Y0 = randn(size(av));
% using the identity
	Y0 = eye(size(Ss(:,:,1)))+1e-1*randn(size(Ss(:,:,1)));
