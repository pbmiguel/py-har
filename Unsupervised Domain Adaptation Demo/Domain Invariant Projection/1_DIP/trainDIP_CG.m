function W = trainDIP_CG(y, Xs, Xt, sigma, lambda, d)

D = size(Xs,2);

global FParameters
FParameters = [];
FParameters.Xs = Xs;
FParameters.Xt = Xt;
FParameters.D = D;
FParameters.d = d;
FParameters.sigma = sigma;
FParameters.ys = y;
FParameters.lambda = lambda;

Y = eye(D);
Y = Y(:,1:d);

global SGParameters;
SGParameters = [];
SGParameters.verbose = 1;
SGParameters.Mode = 0;
SGParameters.metric = 1;
SGParameters.complex = 0;
SGParameters.motion = 1;
SGParameters.gradtol = 1e-7;
SGParameters.ftol = 1e-10;
SGParameters.partition = num2cell(1:size(Y,2));
SGParameters.dimension = dimension(Y);


W = solve_cg(Y);

end






