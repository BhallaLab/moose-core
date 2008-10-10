//moose

int numNodes = $1
int i

for ( i = 0 ; i < numNodes; i = i + 1 )
	create compartment foo{i}@{i}
	setfield /foo{i} Em {i}
	setfield /foo{i} Cm {10 + i * 10}
end

showfield /foo0 *
if ( numNodes > 0 )
	showfield /foo1 *
end

showfield /foo# Em
showfield /foo# Cm

quit
