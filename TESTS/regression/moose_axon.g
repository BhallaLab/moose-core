// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This demo loads a 501 compartment model of a linear excitable neuron. A square
wave pulse of current injection is given to the first compartment, and activity
is recorded from equally spaced compartments. The plots of membrane potential
from these compartments show propagation of action potentials along the length
of the axon.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include axon/compatibility.g
int USE_SOLVER = 1

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 50e-6
float SIMLENGTH = 0.05
float INJECT = 5e-10
float EREST_ACT = -0.065

// include bulbchan.g
include axon/bulbchan.g

ce /library
	make_Na_mit_usb
	make_K_mit_usb
ce /

//=====================================
//  Create cells
//=====================================
readcell axon/axon.p /axon


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plots

create table /plots/Vm0
call /plots/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Vm0 step_mode 3
addmsg /axon/soma /plots/Vm0 INPUT Vm

create table /plots/Vm100
call /plots/Vm100 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Vm100 step_mode 3
addmsg /axon/c100 /plots/Vm100 INPUT Vm

create table /plots/Vm200
call /plots/Vm200 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Vm200 step_mode 3
addmsg /axon/c200 /plots/Vm200 INPUT Vm

create table /plots/Vm300
call /plots/Vm300 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Vm300 step_mode 3
addmsg /axon/c300 /plots/Vm300 INPUT Vm

create table /plots/Vm400
if ( GENESIS )
	call /plots/Vm400 TABCREATE {SIMLENGTH / IODT + 10} 0 {SIMLENGTH}
else
	call /plots/Vm400 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
end
setfield /plots/Vm400 step_mode 3
addmsg /axon/c400 /plots/Vm400 INPUT Vm


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
// Varying current injection
create table /inject
call /inject TABCREATE 100 0 {SIMLENGTH}
setfield /inject step_mode 2
setfield /inject stepsize 0
addmsg /inject /axon/soma INJECT output

float current = { INJECT }

// Injection current flips between 0.0 and {INJECT} at regular intervals
int i
for ( i = 0; i <= 100; i = i + 1 )
	if ( { i % 20 } == 0 )
		current = { INJECT - current }
	end
	
	setfield /inject table->table[{i}] { current }
end

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

useclock /inject 0
useclock /plots/#[TYPE=table] 2

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /axon/solve
		setfield /axon/solve \
			path /axon/##[TYPE=symcompartment],/axon/##[TYPE=compartment] \
			chanmode 1
		call /axon/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /axon method "ee"
	end
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time
if ( GENESIS )
	step 10
end

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
openfile {filename} w

writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(400)"
closefile {filename}
tab2file {filename} /plots/Vm400 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to axon*.plot. Each plot is a trace of membrane potential at a
different point along the axon.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
