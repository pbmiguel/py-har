function [g,sigma] = calc_g( indata )
sz = size(indata);
if ndims(sz)~=2
    fprintf(2,'Input data is not 2-d matrix!\n');
    g = 0;
    return
end
n = sz(1);
d = sz(2);
ave_c = mean(indata);
ave = repmat(ave_c,n,1);
sigma = sum( sum( (indata - ave).^2 ))  / n;
g = 1 / (2*sigma);
sigma = sqrt(sigma);
