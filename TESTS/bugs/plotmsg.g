echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DESCRIPTION:
Messages from tables do not get added if MOOSE is compiled using compiler
optimizations (-O3 in g++).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

create compartment cc

create table /plot
call /plot TABCREATE 100 0.0 1.0

// This fails..
addmsg /cc /plot INPUT Vm

// And this too..
addmsg /plot/inputRequest /cc/Vm

//quit
