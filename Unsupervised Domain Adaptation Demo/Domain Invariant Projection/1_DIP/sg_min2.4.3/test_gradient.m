X = rand(10,5);

K = exp(-L2_distance_2(X', X'));


L = diag(sum(K)) - K;

r1 = X'*L*X;
r1*2

r2 = zeros(size(r1));
for i = 1 : 10
    for j = 1 : 10
        xi = X(i,:)';
        xj = X(j,:)';
        r2 =  r2 + K(i,j)* (xi-xj)*(xi-xj)';
    end
end
r2