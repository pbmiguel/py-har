function F = frank(n, k)
%FRANK   Frank matrix - ill-conditioned eigenvalues.
%        F = FRANK(N, K) is the Frank matrix of order N.  It is upper
%        Hessenberg with determinant 1.  K = 0 is the default; if K = 1 the
%        elements are reflected about the anti-diagonal (1,N)--(N,1).
%        The eigenvalues of F may be obtained in terms of the zeros of the
%        Hermite polynomials.  They are positive and occur in reciprocal
%        pairs.  Thus if N is odd, 1 is an eigenvalue.
%        F has FLOOR(N/2) ill-conditioned eigenvalues---the smaller ones.

%        For large N, DET(FRANK(N)') comes out far from 1---see Frank (1958)
%        and Wilkinson (1960) for discussions.
%
%        References:
%        W.L. Frank, Computing eigenvalues of complex matrices by determinant
%           evaluation and by methods of Danilewski and Wielandt, J. Soc.
%           Indust. Appl. Math., 6 (1958), pp. 378-392 (see pp. 385, 388).
%        G.H. Golub and J.H. Wilkinson, Ill-conditioned eigensystems and the
%           computation of the Jordan canonical form, SIAM Review, 18 (1976),
%             pp. 578-619 (Section 13).
%        H. Rutishauser, On test matrices, Programmation en Mathematiques
%           Numeriques, Editions Centre Nat. Recherche Sci., Paris, 165,
%           1966, pp. 349-365.  Section 9.
%        J.H. Wilkinson, Error analysis of floating-point computation,
%           Numer. Math., 2 (1960), pp. 319-340 (Section 8).
%        J.H. Wilkinson, The Algebraic Eigenvalue Problem, Oxford University
%           Press, 1965 (pp. 92-93).
%        The next two references give details of the eigensystem, as does
%        Rutishauser (see above).
%        P.J. Eberlein, A note on the matrices denoted by B_n, SIAM J. Appl.
%           Math., 20 (1971), pp. 87-92.
%        J.M. Varah, A generalization of the Frank matrix, SIAM J. Sci. Stat.
%           Comput., 7 (1986), pp. 835-839.

if nargin == 1, k = 0; end

F = min( ones(n,1)*(1:n), (1:n)'*ones(1,n) );
%   Take upper Hessenberg part.
F = triu(F,-1);
if k == 0
   p=n:-1:1;
   F = F(p,p)';
end
