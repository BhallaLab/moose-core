//moose

int numNodes = $1
int i

for ( i = 0; i < numNodes; i = i + 1 )
	create neutral foo{i}@{i}
end

le

for ( i = 0; i < numNodes; i = i + 1 )
	setfield /foo{i} name bar{i * 10}
end

str name
int node
for ( i = 0; i < numNodes; i = i + 1 )
	node = { getfield /bar{i * 10} node }
	echo i= {i}, node = {node}
end

le

for ( i = 0; i < numNodes; i = i + 1 )
	delete bar{i * 10}
end

echo after deletion:
le

quit
