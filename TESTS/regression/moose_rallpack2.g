// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rallpack 2: Branching cable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include rallpack2/compatibility.g
include rallpack2/util.g
int USE_SOLVER = 1

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
setfield /plots/v1 step_mode 3
addmsg /cell/c1 /plots/v1 INPUT Vm

create table /plots/vn
call /plots/vn TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/vn step_mode 3
addmsg /cell/c{2 ** MAX_DEPTH - 1} /plots/vn INPUT Vm


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {IODT}

useclock /plots/#[TYPE=table] 3

//=====================================
//  Stimulus
//=====================================
setfield /cell/c1 inject {INJECT}

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /cell/solve
		setfield /cell/solve \
			path /cell/##[TYPE=symcompartment],/cell/##[TYPE=compartment] \
			chanmode 1
		call /cell/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /cell method "ee"
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
str filename
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

filename = "test.plot"
// Clear file contents
openfile {filename} w
closefile {filename}

//filename = "branch-0" @ {extension}
openfile {filename} a
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(0)"
closefile {filename}
tab2file {filename} /plots/v1 table

//filename = "branch-x" @ {extension}
openfile {filename} a
writefile {filename} " "
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(x)"
closefile {filename}
tab2file {filename} /plots/vn table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference curves (analytical and from GENESIS) are in files named
*.analytical.plot and *.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
