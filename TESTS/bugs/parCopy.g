// moose

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(SF.net tracker id: 2199729)

This script crashes when run in parallel.

It copies an HHChannel into a compartment 2 times. The compartment is global
as it is placed in /library.

Use:
	mpirun -np 2 moose parCopy.g
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

ce /library
	create HHChannel hh
	create Compartment cc

	copy hh cc/h1
	copy hh cc/h2
ce /
