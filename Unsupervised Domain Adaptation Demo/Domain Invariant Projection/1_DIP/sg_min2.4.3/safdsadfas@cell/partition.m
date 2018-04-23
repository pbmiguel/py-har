function part = partition(Y0)
% PARTITION	makes a guess at the partition of the function F
%	based on the properties of dF at a randomly selected point Y.
%
%	part = partition(Y0);
%
% role	geometrized objective function, determines the feasible subspace
%	of possible search directions for the function F.
	global SGParameters;
	cm = SGParameters.complex;

	for yi = 1:length(Y0)
		[N,P] = size(Y0{yi});
		Y{yi} = randn(N,P)+i*cm*randn(N,P);
	end
	[Y,r] = qr(Y,0);
	A = Y'*dF(Y);
	As = (A+A')/2;
	Aa = (A-A')/2;
	M = (abs(Aa)<1e-7*abs(As));

	partarg =0;
	for yi = 1:length(Y)
		[N,P] = size(Y{yi});
		scorebd = ones(1,P); np=0;
		for j=1:P,
			if (scorebd(j))
				np=np+1;
				part{yi}{np} = find(M{yi}(j,:));
				scorebd(part{yi}{np}) = 0*part{yi}{np};
			end
		end
	end

	Mcomp = 0*M;
	for yi = 1:length(Y),
		for j=1:length(part{yi}),
			Mcomp{yi}(part{yi}{j},part{yi}{j}) = ones(length(part{yi}{j}));
		end
	end

	goodpart = 1;
	for yi = 1:length(Y),
		goodpart = goodpart & (sum(sum(abs(Mcomp{yi}-M{yi})))==0);
	end
	if ~goodpart
		warning('unable to find consistent partition of F.');
		for yi = 1:length(Y),
			[N,P] = size(Y{yi});
			part{yi} = num2cell(1:P);
		end
	end


