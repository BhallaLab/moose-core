//moose

int numNodes = $1
int NX = 3
int NY = 3
int i

for ( i = 0; i < numNodes; i = i + 1 )
	if ( numNodes > 1 )
		create neutral foo{i}@{i}
	else 
		create neutral foo{i}
	end
	createmap table /foo{i} {NX} {NY} -object
end

int j
int k
/*
*/
for ( i = 0; i < numNodes; i = i + 1 )
	for ( j = 0; j < NX; j = j + 1 )
		for ( k = 0; k < NY; k = k + 1 )
			// echo setfield /foo{i}/table[{j * NY + k}] output { i * 100 + j * 10 + k }
			setfield /foo{i}/table[{j * NY + k}] output { i * 100 + j * 10 + k }
		end
	end
end

le

showfield /foo#/table[] output
quit
