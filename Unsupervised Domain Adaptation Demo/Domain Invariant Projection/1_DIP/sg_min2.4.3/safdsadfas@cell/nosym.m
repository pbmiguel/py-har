function H = nosym(Y,H)
% NOSYM	Orthogonal projection to remove the block diagonal components
%	of the tangent vector H corresponding to the block diagonal
%	right symmetries of F(Y).
%
%	H = NOSYM(Y,H)
%
% role	geometrized objective function, necessary to reduce the number of
%	dimensions of the problem and to have a well conditioned hessian.
%	Somewhat analogous to projecting H to a feasible set of search
%	directions.
	global SGParameters;
	part = SGParameters.partition;

	for yi = 1:length(Y)
		vert = Y{yi}'*H{yi};
		for j = 1:length(part{yi}),
			H{yi}(:,part{yi}{j})=H{yi}(:,part{yi}{j})-Y{yi}(:,part{yi}{j})*vert(part{yi}{j},part{yi}{j});
		end
	end


