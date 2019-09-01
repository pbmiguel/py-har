function Y = solve_cg(Yk)

[n, p] = size(Yk);
FYk = dF(Yk);
Gk = FYk - Yk*Yk'*FYk;
Hk  = -Gk;
fprintf('#iter  |   obj\n');
for  k = 1 : 20 
    [U,S, V] = svd(Hk, 'econ');
    theta = diag(S);
    
    [tk, obj(k)] = line_search(Yk, V, U, theta);
    
    Yk1 = Yk*V*diag(cos(theta*tk))*V' + U*diag(sin(theta*tk))*V';
    FYk1 = dF(Yk1);
    Gk1 = FYk1 - Yk1*Yk1'*FYk1;
    tauHk = (-Yk*V*diag(sin(theta*tk)) + U*diag(cos(theta*tk)))*diag(theta)*V';
    tauGk = Gk - (Yk*V*diag(sin(theta*tk)) + U*diag(1 - cos(theta*tk)))*U'*Gk;
    gammak = trace( (Gk1-tauGk)'*Gk1 ) / trace(Gk'*Gk);
    Hk1 = -Gk1 + gammak*tauHk;
    
    if mod(k+1, p*(n-p)) == 0
        Hk1 = -Gk1;
    end
    
    if k>1 && (abs((obj(k-1)-obj(k))/obj(k)) < 5e-4 ||obj(k)>obj(k-1))
        break;
    end      
    
    fprintf('%d\t|\t%g\n', k, obj(k));
    
    Yk = Yk1;
    Hk = Hk1;   
    Gk = Gk1;
end
Y = Yk;
end

function [t, f] = line_search(Y, V, U, ds)
    % linear search for t
    r = 0.5*(sqrt(5)-1);
    xs = 0;
    xe = 1;
    d  = (xe - xs)*r;
    x1 = xe - d;
    x2 = xs + d;
    f1 = func_obj(x1, Y, V, U, ds);
    f2 = func_obj(x2, Y, V, U, ds);
    tau = 0.001;
    while(1)
        d = d*r;
        if(f1<f2)
            xe = x2;
            
            x2 = x1;
            f2 = f1;
            
            x1 = xe -d;
            f1 = func_obj(x1, Y, V, U, ds);
        else
            xs = x1;
            
            x1 = x2;
            f1 = f2;
            
            x2 = xs + d;
            f2 = func_obj(x2, Y, V, U, ds);
        end
        if(abs(x1 - x2) < tau)
            break;
        end
    end
    if(f1 > f2)
        f = f1;
        t = x1;
    else
        f = f2;
        t = x2;
    end
    
end

function obj = func_obj(t, Y, V, U, theta)

    Yt = Y*V*diag(cos(theta*t))*V' + U*diag(sin(theta*t))*V';
    obj = F(Yt);

end
