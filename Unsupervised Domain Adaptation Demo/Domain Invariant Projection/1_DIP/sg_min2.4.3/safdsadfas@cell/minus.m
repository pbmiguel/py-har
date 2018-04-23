function z = minus(x,y)
if iscell(x) & iscell(y)
for i=1:length(x)
	z{i} = x{i}-y{i};
end
elseif iscell(x) & isnumeric(y)
for i=1:length(x)
	z{i} = x{i}-y;
end
elseif isnumeric(x) & iscell(y)
for i=1:length(y)
	z{i} = x-y{i};
end
end
