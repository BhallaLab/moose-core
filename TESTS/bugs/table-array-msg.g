echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116512)

DESCRIPTION:
Messages from Table to Compartment get added if the compartment is a SimpleElement.
If it is an ArrayElement, or if it is child of an ArrayElement, then the add fails.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

ce /library
create Compartment compt

create Neutral cell
create Compartment cell/cc
ce /

createmap /library/compt / 1 3
createmap /library/cell / 1 3

create Table /table
create Compartment /cc

// OK
addmsg /table/inputRequest /cc/Vm

// Fails: Array of Compartments
addmsg /table/inputRequest /compt[1]/Vm

// Fails: Array of Neutrals containing a Compartment
addmsg /table/inputRequest /cell[1]/cc/Vm

quit
