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

include bulbchan.g

ce /library
	make_Na_mit_usb
	make_K_mit_usb
ce /

//=====================================
//  Create cells
//=====================================
readcell axon.p /axon


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /data

create table /data/Vm0
call /data/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm0 step_mode 3
addmsg /axon/soma /data/Vm0 INPUT Vm

create table /data/Vm100
call /data/Vm100 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm100 step_mode 3
addmsg /axon/c100 /data/Vm100 INPUT Vm

create table /data/Vm200
call /data/Vm200 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm200 step_mode 3
addmsg /axon/c200 /data/Vm200 INPUT Vm

create table /data/Vm300
call /data/Vm300 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm300 step_mode 3
addmsg /axon/c300 /data/Vm300 INPUT Vm

create table /data/Vm400
call /data/Vm400 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm400 step_mode 3
addmsg /axon/c400 /data/Vm400 INPUT Vm


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

filename = "axon0" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(0)"
closefile {filename}
tab2file {filename} /data/Vm0 table

filename = "axon100" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(100)"
closefile {filename}
tab2file {filename} /data/Vm100 table

filename = "axon200" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(200)"
closefile {filename}
tab2file {filename} /data/Vm200 table

filename = "axon300" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(300)"
closefile {filename}
tab2file {filename} /data/Vm300 table

filename = "axon400" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm(400)"
closefile {filename}
tab2file {filename} /data/Vm400 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to axon*.plot. Each plot is a trace of membrane potential at a
different point along the axon.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
