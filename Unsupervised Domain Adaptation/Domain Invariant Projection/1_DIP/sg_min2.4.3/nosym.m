function H = nosym(Y,H)
% NOSYM	Orthogonal projection to remove the block diagonal components
%	of the tangent vector H corresponding to the block diagonal
%	right symmetries of F(Y).
%
%	H = NOSYM(Y,H)
%	Y is expected to satisfy Y'*Y=I
%	H is expected to and sill satisfy H = tangent(Y,H0)
%
% role	geometrized objective function, necessary to reduce the number of
%	dimensions of the problem and to have a well conditioned hessian.
%	Somewhat analogous to projecting H to a feasible set of search
%	directions.
	global SGParameters;
	part = SGParameters.partition;
	vert = Y'*H;
	for j = 1:length(part),
		H(:,part{j})=H(:,part{j})-Y(:,part{j})*vert(part{j},part{j});
	end
