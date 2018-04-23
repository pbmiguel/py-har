function part = partition(Y0)
% PARTITION	makes a guess at the partition of the function F
%	based on the properties of dF at a randomly selected point Y.
%
%	part = partition(Y0);
%	Y0 is expected to satisfy Y0'*Y0 = I
%	P = size(Y0,2)
%
% role	geometrized objective function, determines the feasible subspace
%	of possible search directions for the function F.
	global SGParameters;
	cm = SGParameters.complex;
	[N,P] = size(Y0);
	[Y,r] = qr(randn(N,P)+i*cm*randn(N,P),0);
	A = Y'*dF(Y);
	As = (A+A')/2;
	Aa = (A-A')/2;	
	M = (abs(Aa)<1e-7*abs(As));

	scorebd = ones(1,P); np=0;
	for j=1:P,
		if (scorebd(j))
			np=np+1;
			part{np} = find(M(j,:));
			scorebd(part{np}) = 0*part{np};
		end
	end

	Mcomp = 0*M;
	for j=1:length(part),
		Mcomp(part{j},part{j}) = ones(length(part{j}));
	end

	goodpart = (sum(sum(abs(Mcomp-M)))==0);
	if ~goodpart
		warning('unable to find consistent partition of F.');
		part = num2cell(1:P);
	end
