function tf = isreal(Y)
tf = 1;
for yi = 1:length(Y)
	tf = tf & isreal(Y{yi});
end
