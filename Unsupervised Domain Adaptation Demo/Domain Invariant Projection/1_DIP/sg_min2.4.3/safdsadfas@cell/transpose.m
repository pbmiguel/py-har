function z = transpose(x)
for i = 1:length(x),
z{i} = x{i}.';
end
