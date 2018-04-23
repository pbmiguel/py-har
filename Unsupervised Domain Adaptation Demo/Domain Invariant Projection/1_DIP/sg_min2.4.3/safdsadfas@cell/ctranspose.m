function z = ctranspose(x)
for i = 1:length(x),
z{i} = x{i}';
end
