//moose

// This test examines how the current working element (cwe) is handled
// in parallel.

//
// Need a way to get numNodes from the system.
int numNodes = 2
if ( numNodes > 1 )
	create neutral foo@1
else
	create neutral foo
end

ce foo
create neutral bar

le
le /foo
pwe
ce bar // This dumps.

