// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Same as the Axon demo, except that no channels exist. Also, 2 identical
copies of the cell are created. This is helpful for comparing propagation
in 2 cells with different passive parameters.

This demo loads a 501 compartment model of a linear passive neuron. A square
wave pulse of current injection is given to the first compartment, and activity
is recorded from equally spaced compartments. The plots of membrane potential
from these compartments show propagation of action potentials along the length
of the axon.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g
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
// 
// ce /library
// 	make_Na_mit_usb
// 	make_K_mit_usb
// ce /

//=====================================
//  Create cells
//=====================================
readcell axon-passive.p /axon0
readcell axon-passive.p /axon1


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /data

// axon0
create table /data/Vm0_0
call /data/Vm0_0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm0_0 step_mode 3
addmsg /axon0/soma /data/Vm0_0 INPUT Vm

create table /data/Vm100_0
call /data/Vm100_0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm100_0 step_mode 3
addmsg /axon0/c100 /data/Vm100_0 INPUT Vm

// axon1
create table /data/Vm0_1
call /data/Vm0_1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm0_1 step_mode 3
addmsg /axon1/soma /data/Vm0_1 INPUT Vm

create table /data/Vm100_1
call /data/Vm100_1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm100_1 step_mode 3
addmsg /axon1/c100 /data/Vm100_1 INPUT Vm


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
addmsg /inject /axon0/soma INJECT output
addmsg /inject /axon1/soma INJECT output

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
useclock /data/#[TYPE=table] 2

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /axon0/solve
		setfield /axon0/solve \
			path /axon0/##[TYPE=symcompartment],/axon0/##[TYPE=compartment] \
			chanmode 1
		call /axon0/solve SETUP
		
		create hsolve /axon1/solve
		setfield /axon1/solve \
			path /axon1/##[TYPE=symcompartment],/axon1/##[TYPE=compartment] \
			chanmode 1
		call /axon1/solve SETUP
		
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /axon0 method "ee"
		setfield /axon1 method "ee"
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
/*
str filename
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

filename = "axon0_0" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(0)"
closefile {filename}
tab2file {filename} /data/Vm0_0 table

filename = "axon100_0" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(100)"
closefile {filename}
tab2file {filename} /data/Vm100_0 table

filename = "axon0_1" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(0)"
closefile {filename}
tab2file {filename} /data/Vm0_1 table

filename = "axon100_1" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(100)"
closefile {filename}
tab2file {filename} /data/Vm100_1 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to axon*.plot. Each plot is a trace of membrane potential at a
different point along the axon.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
*/

quit
