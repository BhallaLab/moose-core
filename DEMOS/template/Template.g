// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Template for creating GENESIS- and MOOSE-compatible neuronal models.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

////////////////////////////////////////////////////////////////////////////////
// COMPATIBILITY (between MOOSE and GENESIS)
////////////////////////////////////////////////////////////////////////////////
include compatibility.g


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 100e-6
float SIMLENGTH = 0.05
float INJECT = 1.0e-10
float EREST_ACT = -0.065

include simplechan.g

ce /library
	make_Na_mit_usb
	make_K_mit_usb
ce /

//=====================================
//  Create cells
//=====================================
readcell myelin2.p /axon
readcell cable.p /cable

//=====================================
//  Create synapse
//=====================================
ce /axon/n99/i20
create spikegen spike
setfield spike thresh 0.0 \
               abs_refract .02
addmsg . spike INPUT Vm
ce /

ce /cable/c1
create synchan syn
setfield ^ Ek 0 gmax 1e-8 tau1 1e-3 tau2 2e-3
addmsg . syn VOLTAGE Vm
addmsg syn . CHANNEL Gk Ek
ce /

addmsg /axon/n99/i20/spike /cable/c1/syn SPIKE


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create table /plot
call /plot TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
useclock /plot 2
setfield /plot step_mode 3

//=====================================
//  Record from compartment
//=====================================
addmsg /cable/c10 /plot INPUT Vm


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

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
//  Stimulus
//=====================================
setfield /axon/soma inject {INJECT}

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /axon/solve
	setfield /axon/solve \
		path /axon/##[TYPE=symcompartment],/axon/##[TYPE=compartment] \
		comptmode 1  \
		chanmode 3
	call /axon/solve SETUP
	setmethod 11
	
	create hsolve /cable/solve
	setfield /cable/solve \
		path /cable/##[TYPE=symcompartment],/cable/##[TYPE=compartment] \
		comptmode 1  \
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
openfile "test.plot" w
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"
tab2file test.plot /plot table


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference plot is included. Present curve is in test.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
