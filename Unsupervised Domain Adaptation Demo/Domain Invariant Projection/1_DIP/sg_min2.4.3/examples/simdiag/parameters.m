function parameters(varargin)
% PARAMETERS    initializes the parameters for an instance of a 
%               INDSCAL minimization problem.
%
%       PARAMETERS(S1,S2,...,Sn)
%       Si are expected to be symmetric matrices of equal size.
%
% role  sets up the global parameters used at all levels of computation.
 
        global FParameters;
        FParameters = [];
        FParameters.Ss = cat(3,varargin{:});
