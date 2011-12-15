// moose
// genesis

str script = $0
str model = $1
int arg_present = ( { strcmp { model } "" } != 0 )
if ( ! arg_present )
	echo "Usage: [ moose | genesis ] "{ script }" <model>"
	echo "Where <model> is one of ( passive, reference, A0, A1, B0, B1, C0, C1, D0, D1 )."
	exit
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Testing multiple Ca-pools per compartment using variants of Traub's
1991 Hippocampal CA3 pyramidal cell model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g
int USE_SOLVER = 1

/*
 * Genesis integrates the calcium current (into the calcium pool) in a 
 * slightly different way from Moose. While the integration in Moose is 
 * slightly more accurate, this flag forces Moose to imitate the Genesis 
 * method, to get a better match.
 */
int IMITATE_GENESIS = 1
str OUTPUT_DIR = "output/"

str plots_location = "/data"
str apical = "/CA3/apical_11"
str basal  = "/CA3/basal_7"

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 10e-6
float IODT = 100e-6
float SIMLENGTH = 0.10
float INJECT = 2.0e-10
float EREST_ACT = -0.060
float ENA = 0.115 + EREST_ACT // 0.055  when EREST_ACT = -0.060
float EK = -0.015 + EREST_ACT // -0.075
float ECA = 0.140 + EREST_ACT // 0.080

include traub91proto.g

ce /library
	// Ca channel
	make_Ca_1
	make_Ca_2
	
	// Ca conc
	make_Co_1
	make_Co_2
	make_Co_3
	make_Co_4
	
	// K_AHP: Ca-dependent
	make_K_1_1
	make_K_1_2
	
	make_K_2_1
	
	make_K_3_1
	make_K_3_2
	
	make_K_4_1
	
	// Other channels
	make_Na
	make_K_DR
	make_K_C
	make_K_A
ce /

//=====================================
//  Create cells
//=====================================
readcell models/CA3_{model}.p /CA3

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral {plots_location}

function record( from, to, field )
	str from
	str to
	str field
	
	create table {to}
	call {to} TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield {to} step_mode 3
	
	addmsg {from} {to} INPUT {field}
end

// Helper wrapper function.
function record_w( from, obj, field )
	str from
	str obj
	str field
	
	str tag
	if ( {strcmp {from} "apical"} == 0 )
		from  = {apical}
		tag   = "a"
	elif ( {strcmp {from} "basal"} == 0 )
		from  = {basal}
		tag   = "b"
	else
		echo "Error: record_w(): 'from' should be 'apical' or 'basal'."
		echo "Check script. Aborting."
		exit
	end
	
	str nick
	if ( {strcmp {obj} "."} == 0 )
		nick = "Vm"
	else
		nick = {obj}
		from = {from} @ "/" @ {obj}
	end
	
	str table = {model} @ "_" @ {tag} @ "_" @ {nick}
	str to = {plots_location} @ "/" @ {table}
	
	record {from} {to} {field}
end

record_w "apical" "." "Vm"
record_w "basal"  "." "Vm"

if ( {strcmp {model} "passive"} == 0 )
	// Do nothing
elif ( {strcmp {model} "reference"} == 0 )
	record_w "apical" "Co_1"  "Ca"
	record_w "apical" "K_1_1" "Z"
	
	record_w "basal"  "Co_1"  "Ca"
	record_w "basal"  "K_1_1" "Z"
elif ( {strcmp {model} "A0"} == 0  || {strcmp {model} "A1"} == 0 )
	record_w "apical" "Co_1"  "Ca"
	record_w "apical" "K_1_1" "Z"
	
	record_w "basal"  "Co_1"  "Ca"
	record_w "basal"  "K_1_1" "Z"
	record_w "basal"  "Co_4"  "Ca"
	record_w "basal"  "K_4_1" "Z"
elif ( {strcmp {model} "B0"} == 0  || {strcmp {model} "B1"} == 0 )
	record_w "apical" "Co_1"  "Ca"
	record_w "apical" "K_1_1" "Z"
	record_w "apical" "Co_2"  "Ca"
	record_w "apical" "K_2_1" "Z"
	
	record_w "basal"  "Co_3"  "Ca"
	record_w "basal"  "K_3_1" "Z"
	record_w "basal"  "K_3_2" "Z"
elif ( {strcmp {model} "C0"} == 0  || {strcmp {model} "C1"} == 0 )
	record_w "apical" "Co_3"  "Ca"
	record_w "apical" "K_3_1" "Z"
	
	record_w "basal"  "Co_1"  "Ca"
	record_w "basal"  "K_1_1" "Z"
	record_w "basal"  "Co_2"  "Ca"
	record_w "basal"  "K_2_1" "Z"
elif ( {strcmp {model} "D0"} == 0  || {strcmp {model} "D1"} == 0 )
	record_w "apical" "Co_1"  "Ca"
	record_w "apical" "K_1_1" "Z"
	record_w "apical" "Co_4"  "Ca"
	record_w "apical" "K_4_1" "Z"
	
	record_w "basal"  "Co_1"  "Ca"
	record_w "basal"  "K_1_1" "Z"
	record_w "basal"  "K_1_2" "Z"
else
	echo "Model should be one of ( passive, reference, A0, A1, B0, B1, C0, C1, D0, D1 )."
	echo "Aborting."
	exit
end

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /CA3/soma inject {INJECT}

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {IODT}

useclock {plots_location}/#[TYPE=table] 3

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /CA3/solve
		setfield /CA3/solve \
			path /CA3/##[TYPE=symcompartment],/CA3/##[TYPE=compartment] \
			chanmode 1
		call /CA3/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /CA3 method "ee"
	end
end

//=====================================
//  Simulation
//=====================================
reset

/*
 * Genesis integrates the calcium current (into the calcium pool) in a 
 * slightly different way from Moose. While the integration in Moose is 
 * slightly more accurate, this flag forces Moose to imitate the Genesis 
 * method, to get a better match.
 */
if ( MOOSE && USE_SOLVER )
	setfield /CA3/solve/integ CaAdvance { 1 - IMITATE_GENESIS }
end
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

str filename
str tab
foreach tab ( {el {plots_location}/##[TYPE=table] } )
	filename = {OUTPUT_DIR} @ {getpath {tab} -tail} @ {extension}
	
	openfile {filename} w
	closefile {filename}
	
	tab2file {filename} {tab} table
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.plot. Reference curves from GENESIS are in files named
*.genesis.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

exit
