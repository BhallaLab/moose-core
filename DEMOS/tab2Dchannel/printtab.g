include MoczydKC.g

make_Moczyd_KC

int i, j
int xdivs = 100
int ydivs = 100

for (i = 0; i <= xdivs; i = i + 1)
	for (j = 0; j <= ydivs; j = j + 1)
		echo { getfield Moczyd_KC X_B->table[{i}][{j}] }" " -n
	end
	echo
end

quit
