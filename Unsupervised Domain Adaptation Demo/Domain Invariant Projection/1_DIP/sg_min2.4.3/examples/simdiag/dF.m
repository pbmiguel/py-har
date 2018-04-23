function df = dF(Y)
% dF    Computes the differential of F, that is,
%       df satisfies real(trace(H'*df)) = d/dx (F(Y+x*H)).
%
%       df = DF(Y)
%       Y is expected to satisfy Y'*Y = I
%       df is the same size as Y
%
% role  objective function, this is the routine called to compute the
%       differential of F.
        global FParameters;
        Ss = FParameters.Ss;
	df = 0;
	for k=1:size(Ss,3),
		S = Ss(:,:,k);
		Com = Y*S-S*Y;
		df = df+(Com*S'-S'*Com);
	end
