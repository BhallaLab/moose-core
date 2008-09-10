include compatibility.g
include util.g

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float  SIMDT           = 50e-6
float  IODT            = {SIMDT} * 1.0
float  SIMLENGTH       = 1.0
int    N_COMPARTMENT   = 10
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

int i
for ( i = 2; i <= {N_COMPARTMENT}; i = i + 1 )
	make_compartment /cable/c{i} {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH} {SYMMETRIC}
	link_compartment /cable/c{i - 1} /cable/c{i} {SYMMETRIC}
end

// Inject table
create table /inject
call /inject TABCREATE 100 0 1
setfield /inject step_mode 2
setfield /inject stepsize 0
addmsg /inject /cable/c1 INJECT output

int i
for ( i = 20; i < 60; i = i + 1 )
	setfield /inject table->table[{i}] 1e-10
end

create table /inject1
call /inject1 TABCREATE 100 0 1
setfield /inject1 step_mode 2
setfield /inject1 stepsize 0
addmsg /inject1 /cable/c1 INJECT output

for ( i = 40; i < 80; i = i + 1 )
	setfield /inject1 table->table[{i}] 1e-10
end

create table /inject2
call /inject2 TABCREATE 100 0 1
setfield /inject2 step_mode 2
setfield /inject2 stepsize 0
addmsg /inject2 /cable/c5 INJECT output

for ( i = 0; i < 100; i = i + 1 )
	setfield /inject2 table->table[{i}] -1e-10
end

// Plot table
create table /plot
call /plot TABCREATE {SIMLENGTH / SIMDT} 0 {SIMLENGTH}
setfield /plot step_mode 3
addmsg /cable/c3 /plot INPUT Vm

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
useclock /inject2 0

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
