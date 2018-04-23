function [fn,Yn] = sg_min(Y0,varargin)
% SG_MIN	Stiefel/Grassmann minimization meant for template use.
%
%	[fn,Yn] = SG_MIN(Y0) minimizes F(Y) where F is
%	a matlab function defined in F.m (dF.m and ddF.m),
%	where Y satisfies Y'*Y=I.
%
%	[fn,Yn] = SG_MIN(Y0,rc,mode,metric,verbose,ftol,gradtol,partition)
%
%       Required Argument:  Y0 expected orthonormal, Y0'*Y0 = I
%       Optional Arguments: (may be specified in nearly any order)
%	   rc={'real','complex'} specifies real vs complex computation.
%          mode={'dog','prcg','frcg','newton'} picks search direction:
%	     Dog-Leg steps   (A dog-leg step method)
%	     Polak-Ribiere    Nonlinear Conjugate Gradient
%	     Fletcher-Reeves  Nonlinear Conjugate Gradient
%            Newton's method (Linear Hessian inverter on tangent space)
%	   metric={'flat','euclidean', 'canonical'} 
%	   motion={'approximate','exact'}
%	   verbose={'verbose','quiet'} 
%          ftol   ={ first  of any scalar arguments } convergence tolerance
%	   gradtol={ second of any scalar arguments } convergence tolerance 
%		either convergence tolerance condition is disabled by 
%		explicitly setting its parameter to 0.
%	   partition = cell array describing symmetries in F
%	Defaults: 
%	   rc: 'real' if isreal(Y0), 'complex' otherwise
%	   partition: automatically determined
%	   SG_MIN(Y0,rc,'newton','euclidean','approximate','verbose',...
%		1e-10, 1e-7, partition)
%	Output: 
%	   fn = function minimum
%	   Yn = minimizing argument will satisfy Yn'*Yn=I.
% role	parses the arguments, sets the global parameters and calls the
%	minimizers

        if ~ exist('F','file'),  error('F.m must be in matlab''s path'),   end
        if ~ exist('dF','file'), error('dF.m must be in matlab''s path'),  end
        if ~ exist('ddF','file'),error('ddF.m must be in matlab''s path'), end

	global SGParameters;
	SGParameters = [];

	[Y0,r] = qr(Y0,0);
	SGParameters.verbose = 1;

	nas = length(varargin);
	metarg = 0; rcarg = 0; partarg = 0; ftolarg = 0; gradtolarg = 0;
	mdarg = 0; motarg = 0;
	for j=1:nas,
		if (ischar(varargin{j}))
			if (strcmp(lower(varargin{j}),'approximate'))
				SGParameters.motion = 0; motarg=1;
			elseif (strcmp(lower(varargin{j}),'exact'))
				SGParameters.motion = 1; motarg=1;
			elseif (strcmp(lower(varargin{j}),'flat'))
				SGParameters.metric = 0; metarg=1;
			elseif (strcmp(lower(varargin{j}),'euclidean'))
				SGParameters.metric = 1; metarg=1;
			elseif (strcmp(lower(varargin{j}),'canonical'))
				SGParameters.metric = 2; metarg=1;
			elseif (strcmp(lower(varargin{j}),'real'))
				SGParameters.complex = 0; rcarg=1;
			elseif (strcmp(lower(varargin{j}),'complex'))
				SGParameters.complex = 1; rcarg=1;
			elseif (strcmp(lower(varargin{j}),'quiet'))
				SGParameters.verbose = 0; verbarg=1;
			elseif (strcmp(lower(varargin{j}),'verbose'))
				SGParameters.verbose = 1; verbarg=1;
			elseif (strcmp(lower(varargin{j}),'dog'))
				SGParameters.Mode = 3; mdarg=1;
			elseif (strcmp(lower(varargin{j}),'prcg'))
				SGParameters.Mode = 2; mdarg=1;
			elseif (strcmp(lower(varargin{j}),'frcg'))
				SGParameters.Mode = 1; mdarg=1;
			elseif (strcmp(lower(varargin{j}),'newton'))
				SGParameters.Mode = 0; mdarg=1;
			end
		elseif (iscell(varargin{j}))
			part = varargin{j}; partarg=1;
		elseif (isnumeric(varargin{j}))
			if (ftolarg)
				SGParameters.gradtol = varargin{j}(1); 
				gradtolarg=1;
			else
				SGParameters.ftol = varargin{j}(1);
				ftolarg=1;
			end
		end
	end
% SGParamters.complex = 0 for real and 1 for complex.
	if (rcarg)
		if (~isreal(Y0)) & (1-SGParameters.complex)
			warning('Y0 has imaginary part, but real computation has been declared.  Good luck.');
		end
	else
		SGParameters.complex = ~isreal(Y0);
	end
% SGParameters.metric = 0 for flat, 1 for euclidean, and 2 for canonical
	if (~metarg)
		SGParameters.metric = 1;
	end
% SGParameters.motion = 0 for approximate, 1 for exact
	if (~motarg)
		SGParameters.motion = 0;
	end
	if (~gradtolarg)
		SGParameters.gradtol = 1e-7;
	end
	if (~ftolarg)
		SGParameters.ftol = 1e-10;
	end
	if (~mdarg)
		SGParameters.Mode = 0;
	end
% Make a partition using a possible given one given one
	if (partarg)
		SGParameters.partition = part;
	else
		SGParameters.partition = partition(Y0);
	end

	SGParameters.dimension = dimension(Y0);

	if (SGParameters.Mode == 3)
		[fn,Yn] = sg_dog(Y0);
	elseif (SGParameters.Mode == 2)
		[fn,Yn] = sg_prcg(Y0);
	elseif (SGParameters.Mode == 1)
		[fn,Yn] = sg_frcg(Y0);
	else
		[fn,Yn] = sg_newton(Y0);
	end
