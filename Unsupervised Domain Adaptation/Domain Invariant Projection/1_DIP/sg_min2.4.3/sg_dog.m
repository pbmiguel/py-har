function [fn,Yn]= sg_dog(Y)
% SG_DOG(Y)	Optimize the objective function, F(Y) over all
%	Y such that Y'*Y = I.  Employs a local iterative search with
%	initial point Y and terminates if the magnitude of the gradient
%	falls to gradtol*(initial gradient magnitude) or if the relative
%	decrease in F after some iteration is less than ftol.
%
%	[fn,Yn]= SG_DOG(Y)
%	Y is expected to satisfy Y'*Y = I.
%	Yn will satisfy Yn'*Yn = I.
% role	high level algorithm, dog-leg step Newton's Method.  That is,
%	solving dir = (tI+Hess)\Grad, where t is halved on good steps
%	and doubled on bad ones.
global SGParameters;
gradtol = SGParameters.gradtol;
ftol = SGParameters.ftol;

if (SGParameters.verbose)
    global SGdata;
    SGdata=[];
end
g = grad(Y); mag = sqrt(ip(Y,g,g));
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

delta = mag; delta_max = mag/sqrt(eps); delta_min = mag*sqrt(eps);
while (mag>geps) || (abs(oldf/f-1)>feps)
    N= N+1;
    
    gradsat = (mag<=geps);
    fsat = (abs(oldf/f-1)<=feps);
    if (fsat) fun = 'gradline'; else fun = 'Fline'; end
    
    dr = invdgrad_MINRES(Y,-g,gradtol*oldmag,delta);
    gdr = ip(Y,dr,g); drHdr = ip(Y,dr,dgrad(Y,dr));
    a = -gdr/drHdr;
    a = fminbnd(fun,-2*abs(a),2*abs(a),[],Y,dr);
    Ya = move(Y,dr,a);
    b = -f/gdr;
    b = fminbnd(fun,-2*abs(b),2*abs(b),[],Y,dr);
    Yb = move(Y,dr,b);
    
    if (fsat)
        ga = grad(Ya);
        gb = grad(Yb);
        maga = sqrt(ip(Ya,ga,ga)); magb = sqrt(ip(Yb,gb,gb));
        if (maga<magb) mag = maga; Y = Ya; g = ga; t= a;
        else mag = magb; Y = Yb; g = gb; t= b; end
        newf = F(Y);
    else
        fa = F(Ya);
        fb = F(Yb);
        if (fa<fb) newf = fa; Y = Ya; t = a;
        else newf= fb; Y = Yb; t=b; end
        g = grad(Y); mag = sqrt(ip(Y,g,g));
    end
    pref = f + t*gdr + t^2*drHdr/2;
    
    steptype = 'stay';
    rat = (pref-f)/(newf-f);
    % here is the place where you fudge around with the parameters
    % for the dog-leg algorithm.
    if (0.66<rat & rat<1.5)
        delta=delta/3; steptype='good dog'; end
    if (rat<.25 | 4 < rat)
        delta=delta*4; steptype='bad dog'; end
    delta = min(delta_max,max(delta,delta_min));
    
    oldf = f; f = newf;
    mag=sqrt(ip(Y,g,g));
    if (SGParameters.verbose)
        SGdata(N+1,:) = [N mag f];
        disp(sprintf('%d\t%e\t%e\t%s',N,mag,f,steptype));
    end
end

fn = f;
Yn = Y;