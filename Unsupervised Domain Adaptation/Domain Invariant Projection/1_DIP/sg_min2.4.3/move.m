function [Yo,Ho] = move(Yi,Hi,t)
% MOVE	moves the point Yi and direction Hi along a geodesic of length t
%	in the metric to a point Yo and a direction Ho.
%
%	[Yo,Ho] = MOVE(Yi,Hi,t)
%	Yi is expected to satisfy Yi'*Yi = I
%	Hi is expected to satisfy Hi = tangent(Yi,Hi0)
%	Yo will satisfy Yo'*Yo = I
%	Ho will satisfy Ho = tangent(Yo,Ho0)
%
% role	geometric implementation, the analog to Yo = Yi+Hi*t, Ho = Hi.
global SGParameters;
met = SGParameters.metric;
mot = SGParameters.motion;
[n,k] = size(Yi);

if (t==0)
    Yo = Yi;
    if (nargout==2)
        Ho = Hi;
    end
    return;
end

if (nargout==2)
    mag1 = sqrt(ip(Yi,Hi,Hi));
end
if (met==0)
    % Move by straight lines with a qr projection back to the manifold
    if (nargout==2)
        mag1 = sqrt(ip(Yi,Hi,Hi));
    end
    [Yo,r] = qr(Yi+t*Hi,0);
    if (nargout==2)
        Ho = Hi;
    end
elseif (met==1)
    if (mot==0)
        % This section computes approximate euclidean geodesics
        % using polar decomposition, though just clamping is more
        % efficient and as accurate for short distances.
        if (nargout==2)
            mag1 = sqrt(ip(Yi,Hi,Hi));
        end
        [U,S,V] = svd(Yi+t*Hi,0);
        Yo = U*V';
        if (nargout==2)
            Ho = Hi;
        end
    else
        % This section computes exact euclidean geodesics
        if (nargout==2)
            mag1 = sqrt(ip(Yi,Hi,Hi));
        end
        a = Yi'*Hi;
        [q,r] = qr(Hi-Yi*a,0);
        mn = expm(t*[2*a, -r'; r, zeros(k)]);
        nm = expm(-t*a);
        Yo = (Yi*mn(1:k,1:k) + q*mn(k+1:2*k,1:k))*nm;
        if (nargout==2)
            q =(Yi*mn(1:k,k+1:2*k)+q*mn(k+1:2*k,k+1:2*k));
            Ho = Yi*a+q*r*nm;
        end
    end
elseif (met==2)
    if (mot==0)
        % this section computes approximate canonical geodesics using approximate
        % matrix inverse.
        if (nargout==2)
            mag1 = sqrt(ip(Yi,Hi,Hi));
        end
        a = Yi'*Hi;
        [q,r] = qr(Hi-Yi*a,0);
        geo = t*[a, -r'; r, zeros(k)];
        mn = (eye(2*k)+geo/2)/(eye(2*k)-geo/2);
        mn = mn(:,1:k);
        Yo = Yi*mn(1:k,:) + q*mn(k+1:2*k,:);
        if (nargout==2)
            Ho = Hi*mn(1:k,:) - Yi*(r'*mn(k+1:2*k,:));
        end
    else
        % This section computes exact canonical geodesics
        if (nargout==2)
            mag1 = sqrt(ip(Yi,Hi,Hi));
        end
        a = Yi'*Hi;
        [q,r] = qr(Hi-Yi*a,0);
        geo = t*[a, -r'; r, zeros(k)];
        mn = expm(geo);
        mn = mn(:,1:k);
        Yo = Yi*mn(1:k,:) + q*mn(k+1:2*k,:);
        if (nargout==2)
            Ho = Hi*mn(1:k,:) - Yi*(r'*mn(k+1:2*k,:));
        end
    end
end
% the clamping here does not need to be called very often.
if (nargout==1)
    Yo = clamp(Yo);
elseif (nargout==2)
    [Yo,Ho] = clamp(Yo,Ho);
    mag2 = sqrt(ip(Yo,Ho,Ho));
    Ho = Ho*mag1/mag2;
end
