// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rallpack 2: Branching cable
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
int    MAX_DEPTH       = 10
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01
float  EM              = -0.065
float  INJECT          = 0.1e-9
float  DIAMETER_0      = 16e-6
float  LENGTH_0        = 32e-6
int    SYMMETRIC       = 1

//=====================================
//  Create cells
//=====================================
float diameter = {DIAMETER_0}
float length   = {LENGTH_0}
int   label    = 1

if ( MOOSE )
	create Cell /cell
else
	create neutral /cell
end

make_compartment /cell/c{label} \
	{RA} {RM} {CM} {EM} {0.0} {diameter} {length} {SYMMETRIC}

int i, j
for ( i = 2; i <= MAX_DEPTH; i = i + 1 )
	diameter = {diameter / 2.0 ** (2.0 / 3.0)}
	length   = {length   / 2.0 ** (1.0 / 3.0)}
	
	for ( j = 1; j <= 2 ** (i - 1); j = j + 1 )
		label = label + 1
		make_compartment /cell/c{label} \
			{RA} {RM} {CM} {EM} 0.0 {diameter} {length} {SYMMETRIC}
		link_compartment /cell/c{label / 2} /cell/c{label} {SYMMETRIC}
	end
end

echo "Rallpack 2 model set up."


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plots

create table /plots/v1
call /plots/v1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
useclock /plots/v1 2
setfield /plots/v1 step_mode 3
addmsg /cell/c1 /plots/v1 INPUT Vm

create table /plots/vn
call /plots/vn TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
useclock /plots/vn 2
setfield /plots/vn step_mode 3
addmsg /cell/c{2 ** MAX_DEPTH - 1} /plots/vn INPUT Vm


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /cell/c1023 inject {INJECT}

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

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /cell/solve
	setfield /cell/solve \
		path /cell/##[TYPE=symcompartment],/cell/##[TYPE=compartment] \
		chanmode 3
	call /cell/solve SETUP
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
openfile "sim_branch.0" w
writefile "sim_branch.0" "/newplot"
writefile "sim_branch.0" "/plotname Vm"
closefile "sim_branch.0"
tab2file sim_branch.0 /plots/v1 table

openfile "sim_branch.x" w
writefile "sim_branch.x" "/newplot"
writefile "sim_branch.x" "/plotname Vm"
closefile "sim_branch.x"
tab2file sim_branch.x /plots/vn table


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference plot is included. Present curve is in test.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
