function H = invdgrad_CG(Y,W,tol,dl)
% INVDGRAD_CG  Inverts the operator dgrad.  i.e. solves for H satisfying
%		dl*H+dgrad(Y,H) = W.  The parameter, dl, is used for dogleg
%		steps, and defaults to 0.  Uses a CG algorithm with
%		a tolerance of gep*norm(W), or a tolerance of tol if given.
%		if tol is given.
%
%	H = INVDGRAD_CG(Y,W,tol,s)
%	Y is expected to satisfy Y'*Y=I
%	W is expected to satisfy W = tangent(Y,W0)
%	H will satisfy H = tangent(Y,H0)
%
% role	geometrized objective function, this is the function called when one
%	wishes to invert the geometric hessian.  The analog is H = Hess\W.
global SGParameters;
dim = SGParameters.dimension;
gep = SGParameters.gradtol;

if (nargin<4)
    dl = 0;
end

% p = reshape(dgrad(Y,reshape(q,n,k)),n*k,1) is the black box which
% applies the hessian to a column vector, q, to produce a column vector p.
% Any iterative inversion algorithm that applies the matrix as a black
% box can be used here.

oldd = 0;
x = 0*W;
r = -W;

rho2 = ip(Y,r,r);
rs = rho2;
if (nargin>=3)
    gepr2 = tol^2;
else
    gepr2 = rho2*gep^2;
end
posdef=1;
cn = -1; Nmax= dim;
reset = 1;
while (posdef & rho2 > gepr2 & cn<Nmax) | reset
    cn = cn+1;
    if (reset)
        d = -r;
        reset=0;
    else
        beta = rho2/oldrho2;
        d = -r+beta*d;
    end
    % application of the hessian.
    if (dl==0)
        Ad = dgrad(Y,d);
    else
        Ad = dl*d+dgrad(Y,d);
    end
    
    dAd = ip(Y,d,Ad);
    % terminate if not positive definite
    if (dAd<=0)
        posdef=0;
    else
        dist = rho2/dAd;
        x = x+dist*d;
        % the clamp call here is crucial, as r is a result of cancellations of
        % constrained vectors, and hence may only poorly satisfy the constraint
        r = r+dist*Ad; r = clamp(Y,r);
        oldrho2 = rho2;
        rho2 = ip(Y,r,r);
    end
end % while
if (cn==0)
    x = W;
    if (abs(dl)>0) x = x/dl; end
end
if (SGParameters.verbose & posdef==0)
    disp('  invdgrad: Hessian not positive definite, CG terminating early');
end
if (SGParameters.verbose & cn==Nmax)
    disp('  invdgrad: max iterations reached inverting the hessian by CG'),
end
H = clamp(Y,x);