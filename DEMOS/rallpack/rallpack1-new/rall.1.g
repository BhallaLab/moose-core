// moose
// genesis


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
float  SIMLENGTH       = 0.25
int    N_COMPARTMENT   = 1000
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
if ( MOOSE )
	create Cell /cable
else
	create neutral /cable
end

make_compartment /cable/c1 {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}

int i
for ( i = 2; i <= {N_COMPARTMENT}; i = i + 1 )
	make_compartment /cable/c{i} {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}
	link_compartment /cable/c{i - 1} /cable/c{i} {SYMMETRIC}
end

echo "Rallpack 1 model set up."


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plots

create table /plots/v1
call /plots/v1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/v1 step_mode 3
addmsg /cable/c1 /plots/v1 INPUT Vm

create table /plots/vn
call /plots/vn TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/vn step_mode 3
addmsg /cable/c{N_COMPARTMENT} /plots/vn INPUT Vm


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /cable/c1 inject {INJECT}

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

useclock /plots/v1 2
useclock /plots/vn 2

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /cable/solve
	setfield /cable/solve \
		path /cable/##[TYPE=symcompartment],/cable/##[TYPE=compartment] \
		chanmode 3
	call /cable/solve SETUP
	setmethod 11
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
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

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference plot is included. Present curve is in test.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
