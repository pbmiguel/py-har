function parameters(A,eigvs,segre,type)
% PARAMETERS	initializes the parameters for an instance of a 
%		minimization problem.  This functions must be executed
%		before any attempts to do any jordan structure minimzations.
%
%	PARAMETERS(A,eigvs,segre,type)
%	A is expected to be a square matrix
%	eigvs is expected to be a 1xp matrix of eigenvalues, assumed to
%	be distinct.
%	segre is expected to be a pxq matrix of segre characteristics. 
%		structures (zero padded).
%	type is expected to be the string 'bundle' or 'orbit' 
%
% role	sets up the global parameters used at all levels of computation.

	global FParameters;
	FParameters = [];
	FParameters.A = A;
	FParameters.eigs = eigvs;
	segre = fix(max(segre,0*segre));
% convert the segre characteristics into weyr characteristics.
	blocks = []; 
	if (size(segre,1) ~= length(eigvs))
		error('length of eigvs must equal number of rows of segre');
	end
	for j=1:size(segre,1),
		for k=1:max(segre(j,:)),
			z = sum(segre(j,:)==k);
			if (z==1)
			disp(sprintf('   %d Jordan block  of order %d of eigenvalue %f',z,k,eigvs(j)));
			end
			if (z>1)
			disp(sprintf('   %d Jordan blocks of order %d of eigenvalue %f',z,k,eigvs(j)));
			end
		end
	end
	for j=1:size(segre,1),
		for k=1:max(segre(j,:)),
			blocks(j,k) = sum(segre(j,:)>=k);
		end
	end	
	FParameters.blocks = blocks;
	if (strcmp(lower(type),'orbit'))
		FParameters.bundle = 0;
	elseif (strcmp(lower(type),'bundle'))
		FParameters.bundle = 1;
	else
		'Either bundle or orbit must be specified.',
	end
