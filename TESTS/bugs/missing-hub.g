echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116499)

DESCRIPTION:
The Kinetic and HSolve managers create an object called 'hub' during reset.
When 'le hub' or 'showmsg hub' is called, MOOSE complains that the element
does not exist.

FURTHER DETAIL:
A correctly set up cell or kinetics model (with a solver) has the following 
element hierarchy:

	Manager
	  L (compartments / molecules / etc.)
	  L solve (Neutral)
		 L hub
		 L integ

The command:
	le solve
shows the hub as a child of solve. However:
	le solve/hub
or even:
	showmsg solve/hub
cause MOOSE to complain that the hub does not exist.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

create Cell /cell
create Compartment /cell/cc

create KineticManager /kinetics

reset

le /cell/solve
le /cell/solve/hub
showmsg /cell/solve/hub

le /kinetics/solve
le /kinetics/solve/hub
showmsg /kinetics/solve/hub

quit
