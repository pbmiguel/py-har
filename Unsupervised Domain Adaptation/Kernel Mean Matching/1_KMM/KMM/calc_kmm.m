function KMM_beta = calc_kmm(Kss, Kst, epsilon, B)
    
    [ns,nt] = size(Kst);
    if nargin<3
        epsilon = 0;
    elseif nargin<4
        B = Inf;
    end
    
    % prepare for solving QP problem
    H = Kss;
    f = sum(Kst,2);
    f = -ns/nt*f';
    
    A1 = [ones(1,ns); -ones(1,ns)];
    b1 = [ns*(1+epsilon); ns*(epsilon-1)];
    
    A2 = [];
    b2 = [];
    
    lb = zeros(ns,1);
    ub = B*ones(ns,1);
    
    KMM_beta  = quadprog(H, f, A1, b1, A2, b2, lb, ub);
    
end