function [fn, Yn] = demo_test2()

global FParameters;
FParameters.A = [
    2    -1     0     0     0     0     0     0     0     0     0     0
    -1     2    -1     0     0     0     0     0     0     0     0     0
    0    -1     2    -1     0     0     0     0     0     0     0     0
    0     0    -1     2    -1     0     0     0     0     0     0     0
    0     0     0    -1     2    -1     0     0     0     0     0     0
    0     0     0     0    -1     2    -1     0     0     0     0     0
    0     0     0     0     0    -1     2    -1     0     0     0     0
    0     0     0     0     0     0    -1     2    -1     0     0     0
    0     0     0     0     0     0     0    -1     2    -1     0     0
    0     0     0     0     0     0     0     0    -1     2    -1     0
    0     0     0     0     0     0     0     0     0    -1     2    -1
    0     0     0     0     0     0     0     0     0     0    -1   2];



Y = [
    -0.1538    0.2465    0.1512   -0.4095
    -0.5923   -0.4383   -0.2529    0.0271
    0.0446    0.0243   -0.3729    0.2063
    0.1023   -0.0774    0.4985   -0.1958
    -0.4077   -0.1657    0.4107    0.3473
    0.4235   -0.1920   -0.0941   -0.2250
    0.4229   -0.1434    0.1879    0.3973
    -0.0134    0.2651   -0.2484   -0.3008
    0.1164   -0.1395   -0.2088    0.3580
    0.0621    0.3913   -0.2662    0.3412
    -0.0664   -0.1956   -0.3617   -0.2246
    0.2581   -0.6121   -0.0761   -0.1899
    ];



global SGParameters;
SGParameters.motion = 1;
gradtol = SGParameters.gradtol;
ftol = SGParameters.ftol;

if (SGParameters.verbose)
    global SGdata;
    SGdata=[];
end
g = grad(Y);
mag = sqrt(ip(Y,g,g)); % the norm of the gradient in the metric space
geps = mag*gradtol;
f = F(Y);
feps = ftol;

N = 0; oldf = 2*f; oldmag = mag;
if (SGParameters.verbose)
    SGdata = [];
    disp(sprintf('%s\t%s\t\t%s\t\t%s','iter','grad','F(Y)','step type'));
    SGdata(N+1,:) = [N mag f];
    disp(sprintf('%d\t%e\t%e\t%s',N,mag,f,'none'));
end

while (mag>geps) || (abs(oldf/f-1)>feps)
    
    N= N+1;
    fun = 'Fline';
    sdr = -g; % nagative gradient
    gsdr = ip(Y,sdr,g); % the negative norm of the gradient
    sa = abs(f/gsdr); % ???
    sa = fminbnd(fun,-3*sa,3*sa,[],Y,sdr);
    Ysa = move(Y,sdr,sa);
    
    sdrHsdr = ip(Y,sdr,dgrad(Y,sdr));
    sb = abs(gsdr/sdrHsdr);
    sb = fminbnd(fun,-2*sb,2*sb,[],Y,sdr);
    Ysb = move(Y,sdr,sb);
    
    if (sb~=sa)
        aa=1;
    end
    
    fsa = F(Ysa);
    fsb = F(Ysb);
    if (SGParameters.verbose)
        steptype='steepest step';
    end
    if (fsa<fsb)
        newf = fsa;
        Y = Ysa;
    else
        newf= fsb;
        Y = Ysb;
    end
    
    g = grad(Y);     
    oldf = f; f = newf;
    mag=sqrt(ip(Y,g,g));
    if (SGParameters.verbose)
        SGdata(N+1,:) = [N mag f];
        disp(sprintf('%d\t%e\t%e\t%s',N,mag,f,steptype));
    end
end

fn = f;
Yn = Y;
