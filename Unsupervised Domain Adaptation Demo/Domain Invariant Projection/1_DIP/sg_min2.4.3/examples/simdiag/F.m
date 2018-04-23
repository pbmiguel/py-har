function f = F(Y)
% F     Computes the energy associated with the stiefel point Y.
%       In this case, by the equation f = sum_i ||Y S_i- S_i Y||^2/2.
%
%       f = F(Y)
%       Y is expected to satisfy Y'*Y=I
%
% role  objective function, the routine called to compute the objective 
%       function.
        global FParameters;
        Ss = FParameters.Ss;
	f = 0;
	for k=1:size(Ss,3),
		S = Ss(:,:,k);
		Com = Y*S-S*Y;
		f = f + sum(sum(  Com.^2 ) )/2;
	end
