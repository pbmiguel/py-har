function [q,r] = qr(A,b)
	if (nargout<2)
		for ai = 1:length(A),
			q{ai} = qr(A{ai});
		end
		return;
	end
	if (nargin<2)
		for ai = 1:length(A),
			[q{ai}, r{ai}] = qr(A{ai});
		end
		return;
	end
	for ai = 1:length(A),
		[q{ai}, r{ai}] = qr(A{ai},b);
	end
