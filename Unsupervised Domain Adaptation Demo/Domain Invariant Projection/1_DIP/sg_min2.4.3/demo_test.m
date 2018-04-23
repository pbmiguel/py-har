function demo_test()
n=5;p=3;
A = randn(n);
B = A*eye(n,p);
Y0 = eye(n,p); % Known solution Y0
H = 0.1*randn(n,p); H=H- Y0*(H'*Y0); % small tangent vector H at Y0
Y = stiefgeod(Y0,H); % Initial guess Y (close to know solution Y0)
% Newton iteration (demonstrate quadratic convergence)
d = norm(Y-Y0,'fro')
while d > sqrt(eps)
    H = procrnt(Y,A,B);
    Y = stiefgeod(Y,H);
    d = norm(Y-Y0,'fro')
end


function [Yt,Ht] = stiefgeod(Y,H,t)
%STIEFGEOD Geodesic on the Stiefel manifold.
%   STIEFGEOD(Y,H) is the geodesic on the Stiefel manifold
%   emanating from Y in direction H, where Y'*Y = eye(p), Y'*H = skew-hermitian,
%   and Y and H are n-by-p matrices.
%
%   STIEFGEOD(Y,H,t) produces the geodesic step in direction H scaled by t.
%   [Yt,Ht] = STIEFGEOD(Y,H,t) produces the geodesic step and the geodesic direction.
[n,p] = size(Y);
if nargin < 3
    t=1;
end
A = Y'*H;
A = (A - A')/2; % Ensure skew-symmetry
[Q,R] = qr(H - Y*A,0);
MN = expm(t*[A,-R';R,zeros(p)]); MN = MN(:,1:p);

Yt = Y*MN(1:p,:) + Q*MN(p+1:2*p,:); % Geodesic from (2.45)
if nargout > 1
    Ht = H*MN(1:p,:) - Y*(R'*MN(p+1:2*p,:));
end % Geodesic direction from (3.3)




function H = procrnt(Y,A,B)
%PROCRNT Newton Step on Stiefel Manifold for 1/2*norm(A*Y-B,'fro')^2.
%   H = PROCRNT(Y,A,B) computes the Newton step on the Stiefel manifold for
%   the function 1/2*norm(A*Y-B,'fro')^2, where Y'*Y = eye(size(Y,2)).

[n,p] = size(Y);
AA = A'*A;
FY = AA*Y - A'*B;
YFY = Y'*FY;
G = FY - Y*YFY';
% Linear conjugate gradient to solve a Newton step
dimV = p*(p-1)/2 + p*(n-p); % == dim Stiefel manifold

% This linear CG code is modified directly from Golub and Van Loan [45]
H = zeros(size(Y)); R1 = -G; P = R1; P0 = zeros(size(Y));
for k=1:dimV
    normR1 = sqrt(stiefip(Y,R1,R1));
    if normR1 < prod(size(Y))*eps
        break;
    end
    if k == 1
        beta = 0;
    else
        beta = (normR1/normR0)^2;
    end
    P0 = P;
    P = R1 + beta*P;
    FYP = FY'*P;
    YP = Y'*P;
    LP = AA*P - Y*(P'*AA*Y) ... % Linear operation on P
        - Y*((FYP-FYP')/2) - (P*YFY'-FY*YP')/2 - (P-Y*YP)*(YFY/2);
    alpha = normR1^2/stiefip(Y,P,LP);
    H=H+ alpha*P;
    R0 = R1;
    normR0 = normR1;
    R1 = R1 - alpha*LP;
end

function ip = stiefip(Y,A,B)
%STIEFIP Inner product (metric) for the Stiefel manifold.
%   ip = STIEFIP(Y,A,B) returns trace(A'*(eye(n)-1/2*Y*Y')*B),
%   where Y'*Y = eye(p), Y'*A & Y'*B = skew-hermitian, and Y, A,
%   and B are n-by-p matrices.
ip = sum(sum(conj(A).*(B - Y*((Y'*B)/2)))); % Canonical metric from (2.39)


