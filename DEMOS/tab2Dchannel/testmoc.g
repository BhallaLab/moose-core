include compatibility.g
include loadMoc.g

create compartment cc
ce cc
	make_Moczyd_KC
	create Ca_concen Ca_conc

	if ( GENESIS )
		addmsg Ca_conc Moczyd_KC CONCEN1 Ca
	else
		addmsg Ca_conc Moczyd_KC CONCEN Ca
	end
ce ..

reset

step 1.0 -t
// Give step of Vm and Ca
step 1.0 -t
// Bring Vm and Ca back to baseline
step 1.0 -t

//quit
