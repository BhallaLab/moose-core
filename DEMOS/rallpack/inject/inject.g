include compatibility.g
include util.g

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float  SIMDT           = 50e-6
float  IODT            = {SIMDT} * 10.0
float  SIMLENGTH       = 1.0
int    N_COMPARTMENT   = 1
float  CABLE_LENGTH    = 1e-3
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01
float  EM              = -0.065
float  INJECT          = 1e-10
float  DIAMETER        = 1e-6
float  LENGTH          = {CABLE_LENGTH} / {N_COMPARTMENT}
int    SYMMETRIC       = 1

if ( MOOSE )
	create Cell /cable
else
	create neutral /cable
end

make_compartment /cable/c1 {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}

// Inject table
create table /inject
call /inject TABCREATE 100 0 1
setfield /inject step_mode 2
setfield /inject stepsize 0
if ( MOOSE )
	addmsg /inject/outputSrc /cable/c1/injectMsg
else
	addmsg /inject /cable/c1 INJECT output
end

int i
for ( i = 20; i < 60; i = i + 1 )
	setfield /inject table->table[{i}] 1e-10
end

create table /inject1
call /inject1 TABCREATE 100 0 1
setfield /inject1 step_mode 2
setfield /inject1 stepsize 0
if ( MOOSE )
	addmsg /inject1/outputSrc /cable/c1/injectMsg
else
	addmsg /inject1 /cable/c1 INJECT output
end

int i
for ( i = 40; i < 80; i = i + 1 )
	setfield /inject1 table->table[{i}] 1e-10
end

// Plot table
create table /plot
call /plot TABCREATE {SIMLENGTH / SIMDT} 0 {SIMLENGTH}
setfield /plot step_mode 3
addmsg /cable/c1 /plot INPUT Vm
if ( MOOSE )
//	addmsg /cable/c1/inject /plot/msgInput
//	addmsg /inject/outputSrc /plot/msgInput
else
//	addmsg /cable/c1 /plot INPUT inject
//	addmsg /inject /plot INPUT output
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

useclock /plot 0
useclock /inject 0
useclock /inject1 0

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /cable/solve
	setfield /cable/solve \
		path /cable/##[TYPE=symcompartment],/cable/##[TYPE=compartment] \
		chanmode 1
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

str filename = "dest.plot"
if ( MOOSE )
	filename = {filename} @ ".moose"
else
	filename = {filename} @ ".genesis"
end
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /plot table

quit
