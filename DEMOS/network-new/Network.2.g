// moose
// genesis

include compatibility.g

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 100e-6
float SIMLENGTH = 0.05
float INJECT = 5.0e-10
float EREST_ACT = -0.065
int N_INPUT = 10
int N_OUTPUT = 10
int i
int j

include hh_tchan.g

ce /library
	make_Na_hh_tchan
	make_K_hh_tchan
	
	//
	//  Prototype compartment
	//
	create compartment compt
	copy Na_hh_tchan compt
	copy K_hh_tchan compt
	setfield compt Rm 1e10 Ra 1e8 Cm 1e-10 initVm -0.065 Em -0.065 Vm 0
	addmsg compt compt/Na_hh_tchan VOLTAGE Vm
	addmsg compt/Na_hh_tchan compt CHANNEL Gk Ek
	addmsg compt compt/K_hh_tchan VOLTAGE Vm
	addmsg compt/K_hh_tchan compt CHANNEL Gk Ek
	
	if ( MOOSE )
		create Cell in_cell
		create Cell out_cell
	else
		create neutral in_cell
		create neutral out_cell
	end
	
	//
	//  Input cell
	//
	copy compt in_cell/in_compt
	
	//
	//  Output cell
	//
	copy compt out_cell/out_compt
ce /

createmap /library/in_cell in_array 1 10
//createmap /library/out_cell out_array 1 10

if ( 1 )
	setfield /in_array/in_cell[] method "ee"
	setfield /out_array/out_cell[] method "ee"
end

if ( MOOSE )
	setclock 0 {SIMDT} 0
	setclock 1 {SIMDT} 1
	setclock 2 {IODT} 0
else
	setclock 0 {SIMDT}
	setclock 1 {SIMDT}
	setclock 2 {IODT}
end

reset
step 10
// step { SIMLENGTH } -t
//quit
