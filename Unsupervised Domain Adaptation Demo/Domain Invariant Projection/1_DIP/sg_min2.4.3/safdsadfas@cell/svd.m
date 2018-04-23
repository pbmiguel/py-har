function [u,s,v] = svd(A,b)
	if (nargout==1)
		for ai = 1:length(A),
			u{ai} = svd(A{ai});
		end
		return;
	end
	if (nargout<3)
		error('Not enough output arguments');
	end
	if (nargin<2)
		for ai = 1:length(A),
			[u{ai}, s{ai}, v{ai}] = svd(A{ai});
		end
		return;
	end
	for ai = 1:length(A),
		[u{ai}, s{ai}, v{ai}] = svd(A{ai},b);
	end
