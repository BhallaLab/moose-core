// moose
// genesis

int NCELL = $1
int SIZE = $2
float SIM_LENGTH = $3
int PLOT = $4

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rallpack 1: Linear cable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g
include util.g

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float  SIMDT           = 50e-6
float  IODT            = {SIMDT} * 1.0
float  SIMLENGTH       = {SIM_LENGTH}
int    N_COMPARTMENT   = {SIZE}
float  CABLE_LENGTH    = 1e-3
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01
float  EM              = -0.065
float  INJECT          = 1e-10
float  DIAMETER        = 1e-6
float  LENGTH          = {CABLE_LENGTH} / {N_COMPARTMENT}
int    SYMMETRIC       = 1

//=====================================
//  Create cells
//=====================================
ce /proto

if ( MOOSE )
	create Cell cable
else
	create neutral cable
end

make_compartment cable/c1 {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}

int i
for ( i = 2; i <= {N_COMPARTMENT}; i = i + 1 )
	make_compartment cable/c{i} {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}
	link_compartment cable/c{i - 1} cable/c{i} {SYMMETRIC}
end

ce /

// SHOULD USE CREATEMAP
int i
for ( i = 1; i <= NCELL; i = i + 1 )
	copy /proto/cable /cable{i}
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
if ( PLOT )
	create neutral /plots

	create table /plots/v1
	call /plots/v1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield /plots/v1 step_mode 3
	addmsg /cable1/c1 /plots/v1 INPUT Vm

	create table /plots/vn
	call /plots/vn TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield /plots/vn step_mode 3
	addmsg /cable1/c{N_COMPARTMENT} /plots/vn INPUT Vm
end

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
for ( i = 1; i <= NCELL; i = i + 1 )
	setfield /cable{i}/c1 inject {INJECT}
end

//=====================================
//  Clocks
//=====================================
if ( MOOSE )
	setclock 0 {SIMDT} 0
	setclock 1 {SIMDT} 1
	setclock 2 {IODT} 0
else
	setclock 0 {SIMDT}
	setclock 1 {SIMDT}
	setclock 2 {IODT}
end

if ( PLOT )
	useclock /plots/v1 2
	useclock /plots/vn 2
end

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	for ( i = 1; i <= NCELL; i = i + 1 )
		create hsolve /cable{i}/solve
		setfield /cable{i}/solve \
			path /cable{i}/##[TYPE=symcompartment],/cable{i}/##[TYPE=compartment] \
			chanmode 3
		call /cable{i}/solve SETUP
		setmethod 11
	end
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
if ( PLOT )
	openfile "sim_cable.0" w
	writefile "sim_cable.0" "/newplot"
	writefile "sim_cable.0" "/plotname Vm"
	closefile "sim_cable.0"
	tab2file "sim_cable.0" /plots/v1 table

	openfile "sim_cable.x" w
	writefile "sim_cable.x" "/newplot"
	writefile "sim_cable.x" "/plotname Vm"
	closefile "sim_cable.x"
	tab2file "sim_cable.x" /plots/vn table
end


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference plot is included. Present curve is in test.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
