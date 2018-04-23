function [Yo,Ho] = move(Yi,Hi,t)
% MOVE	moves the point Yi and direction Hi along a geodesic of length t
%	in the metric to a point Yo and a direction Ho.
%
%	[Yo,Ho] = MOVE(Yi,Hi,t)
%
% role	geometric implementation, the analog to Yo = Yi+Hi*t, Ho = Hi.
	global SGParameters;
	met = SGParameters.metric;
	mot = SGParameters.motion;

	if (t==0) 
		Yo = Yi; 
		if (nargout==2)
			Ho = Hi;
		end
		return;
	end

	if (nargout==2) mag1 = sqrt(ip(Yi,Hi,Hi)); end
	if (met==0)
% Move by straight lines with a qr projection back to the manifold
		if (nargout==2)	mag1 = sqrt(ip(Yi,Hi,Hi)); end
		for yi = 1:length(Yi)
			[Yo{yi},r] = qr(Yi{yi}+t*Hi{yi},0);
		end
	elseif (met==1)
		if (mot==0)
% This section computes approximate euclidean geodesics 
% using polar decomposition
			for yi=1:length(Yi)
				[U,S,V] = svd(Yi{yi}+t*Hi{yi},0); 
				Yo{yi} = U*V';
			end
			if (nargout==2)
				Ho = clamp(Yo,Hi);
				mag2 = sqrt(ip(Yo,Ho,Ho));
				Ho = Ho*mag1/mag2;
			end
		else
% This section computes exact euclidean geodesics
			if (nargout==2) mag1 = sqrt(ip(Yi,Hi,Hi)); end
			for yi = 1:length(Yi)
				k = size(Yi{yi},2);
				a = Yi{yi}'*Hi{yi};
				[q,r] = qr(Hi{yi}-Yi{yi}*a,0);
				mn = expm(t*[2*a, -r'; r, zeros(k)]);
				nm = expm(-t*a);
				Yo{yi} = (Yi{yi}*mn(1:k,1:k) + q*mn(k+1:2*k,1:k))*nm;
				if (nargout==2)
					q =(Yi{yi}*mn(1:k,k+1:2*k)+q*mn(k+1:2*k,k+1:2*k));
					Ho{yi} = Yi{yi}*a+q*r*nm;
				end
			end
		end
	elseif (met==2)
		if (mot==0)
% this section computes approximate canonical geodesics using approximate
% matrix inverse.
			if (nargout==2) mag1 = sqrt(ip(Yi,Hi,Hi)); end
			for yi=1:length(Yi),
				k = size(Yi{yi},2);
				a = Yi{yi}'*Hi{yi};
				[q,r] = qr(Hi{yi}-Yi{yi}*a,0);
				geo = t*[a, -r'; r, zeros(k)];
				mn = (eye(2*k)+geo/2)/(eye(2*k)-geo/2);
				mn = mn(:,1:k);
				Yo{yi} = Yi{yi}*mn(1:k,:) + q*mn(k+1:2*k,:);
				if (nargout==2)
					Ho{yi} = Hi{yi}*mn(1:k,:) - Yi{yi}*(r'*mn(k+1:2*k,:));
				end
			end
		else
% This section computes exact canonical geodesics
			for yi=1:length(Yi),
				k = size(Yi{yi},2);
				a = Yi{yi}'*Hi{yi};
				[q,r] = qr(Hi{yi}-Yi{yi}*a,0);
				geo = t*[a, -r'; r, zeros(k)];
				mn = expm(geo); 
				mn = mn(:,1:k);
				Yo = Yi{yi}*mn(1:k,:) + q*mn(k+1:2*k,:);
				if (nargout==2)
					Ho{yi} = Hi{yi}*mn(1:k,:) - Yi{yi}*(r'*mn(k+1:2*k,:));
				end
			end
		end
	end
	if (nargout==1)
		Yo = clamp(Yo);
	else
		[Yo,Ho] = clamp(Yo,Ho);
		mag2 = sqrt(ip(Yo,Ho,Ho));
		Ho = Ho*mag1/mag2;
	end







