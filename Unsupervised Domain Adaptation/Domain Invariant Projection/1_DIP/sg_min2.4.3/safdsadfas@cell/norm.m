function z = norm(x,arg)
z = 0;
if (nargin<2)
	for i=1:length(x)
		z = z+norm(x{i})^2;
	end
else
	for i=1:length(x)
		z = z+norm(x{i},arg)^2;
	end
end
z = sqrt(z);