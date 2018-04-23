function [H,rs,x,r] = invdgrad_MINRES(Y,W,tol,dl)
% INVDGRAD_MINRES  Inverts the operator dgrad.  i.e. solves for H satisfying
%		dl*H+dgrad(Y,H) = W.  The parameter, dl, is used for dogleg
%		steps, and defaults to 0.  Uses a MINRES algorithm with
%		a tolerance of gep*norm(W), or a tolerance of tol if given.
%		if tol is given.
%
%	H = INVDGRAD_MINRES(Y,W,tol,dl)
%	Y is expected to satisfy Y'*Y=I
%	W is expected to satisfy W = tangent(Y,W0)
%	H will satisfy H = tangent(Y,H0)
%
% role	geometrized objective function, this is the function called when one
%	wishes to invert the geometric hessian.  The analog is
%	H = (dl*I+Hess)\W.
global SGParameters;
dim = SGParameters.dimension;
gep = SGParameters.gradtol;

if (nargin<4)
    dl = 0;
end
x = 0*W;
r = W;
rho = sqrt(ip(Y,r,r)); v = r/rho; rho_old = rho;
if (nargin>=3)
    gepr = tol;
else
    gepr = rho*gep;
end
beta= 0; v_old = 0*v;
beta_t = 0; c = -1; s = 0;
w = 0*v; www = v;

% p = reshape(dgrad(Y,reshape(q,n,k)),n*k,1) is the black box which
% applies the hessian to a column vector, q, to produce a column vector p.
% Any iterative inversion algorithm that applies the matrix as a black
% box can be used here.

cn = -1; Nmax = 2*dim; posdef = 1; reset = 1;
while (rho>gepr & cn<Nmax & posdef & rho <= rho_old*(1+gepr)) | reset
    % application of the hessian.
    cn = cn+1;
    reset = 0;
    if (dl==0)
        wv = dgrad(Y,v)-beta*v_old;
    else
        wv = dl*v+dgrad(Y,v)-beta*v_old;
    end
    alpha = ip(Y,v,wv); wv = wv-alpha*v;
    if (alpha<=0)
        posdef = 0;
    else
        beta = sqrt(ip(Y,wv,wv)); v_old = v; v = wv/beta;
        
        l1 = s*alpha - c*beta_t; l2 = s*beta;
        
        alpha_t = -s*beta_t - c*alpha;  beta_t = c*beta;
        
        l0 = sqrt(alpha_t*alpha_t+beta*beta);
        c = alpha_t/l0; s = beta/l0;
        
        ww = www - l1*w; ww=clamp(Y,ww);	% cancellation line
        www = v - l2*w; w = ww/l0;
        
        x =  x + (rho*c)*w; rho_old = rho; rho =  s*rho;
    end
end %while
if (cn==0)
    x = W;
    if (abs(dl)>0) x = x/dl; end
end

if (SGParameters.verbose & ~posdef)
    disp('  invdgrad: Hessian not positive definite, MINRES terminating early');
end
if (SGParameters.verbose & cn==Nmax)
    disp('  invdgrad: max iterations reached inverting the hessian by MINRES'),
end
if (SGParameters.verbose & rho>rho_old*(1+gepr))
    disp('  invdgrad: residual increase detected in MINRES, terminating early'),
end
H = clamp(Y,x);

