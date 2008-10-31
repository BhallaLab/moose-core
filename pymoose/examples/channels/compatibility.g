// moose
// genesis

//
// Including this script will let other scripts run on MOOSE as well as GENESIS
//

int MOOSE
int GENESIS
if ( {version} < 3.0 )
	MOOSE = 0
	GENESIS = 1
else
	MOOSE = 1
	GENESIS = 0
end

if ( GENESIS )
	create neutral /library
	pushe /library
		create compartment compartment
		create symcompartment symcompartment
	pope
end

//
//  Compatibility with old GENESIS versions
//

addalias setup_table2 setupgate
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau
addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau

function settab2const(gate, table, imin, imax, value)
    str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
		setfield {gate} {table}->table[{i}] {value}
	end
end

function setup_table( gate, table, xdivs, A, B, C, D, F )
	setupgate {gate} {table} {A} {B} {C} {D} {F} -size {xdivs} \
		-range -0.1 0.05
end

function setup_table3(gate, table, xdivs, xmin, xmax, A, B, C, D, F)
	setupgate {gate} {table} {A} {B} {C} {D} {F} -size {xdivs} \
		-range {xmin} {xmax}
end
