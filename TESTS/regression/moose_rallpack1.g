// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rallpack 1: Linear cable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include rallpack1/compatibility.g
include rallpack1/util.g
int USE_SOLVER = 1

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
setfield /cable/c1 inject {INJECT}

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /cable/solve
		setfield /cable/solve \
			path /cable/##[TYPE=symcompartment],/cable/##[TYPE=compartment] \
			chanmode 1
		call /cable/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /cable method "ee"
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

//filename = "cable-0" @ {extension}
openfile {filename} a
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(0)"
closefile {filename}
tab2file {filename} /plots/v1 table

//filename = "cable-x" @ {extension}
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
