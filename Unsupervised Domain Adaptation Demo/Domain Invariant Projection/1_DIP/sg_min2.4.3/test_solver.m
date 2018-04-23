function test_solver
global FParameters
FParameters = [];

FParameters.Xs = rand(10,5);
FParameters.Xt = rand(20,5);
FParameters.D = 5;
FParameters.d = 2;
FParameters.sigma = 1;

Y = orth(rand(5,2));

for i = 1 : 100
    g = grad(Y);
    sdr = -g;
    mag = sqrt(ip(Y, g, g));
    gsdr = ip(Y, sdr, g);
    sb = abs(gsdr);
    [sb, obj(i)] = fminbnd( @func_obj, -2*sb, 2*sb, optimset('Display','off'), Y,  sdr);
    Y = move(Y, sdr, sb);
    fprintf('obj = %g, mag = %g\n', obj(i), mag);
end
plot(obj, '--o');


function obj = func_obj(t, Y, sdr)
Y_new = move(Y, sdr, t);
obj = F(Y_new);
