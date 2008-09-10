// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo loads a 501 compartment model of a linear excitable neuron. Current %
% injection in the first compartment leads to spiking, and the plots show       %
% propagation of action potentials along the length of the axon.                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
float INJECT = 1e-9
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
create table /Vm0
call /Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /Vm0 step_mode 3
addmsg /axon/soma /Vm0 INPUT Vm
useclock /Vm0 2

create table /Vm100
call /Vm100 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /Vm100 step_mode 3
addmsg /axon/c100 /Vm100 INPUT Vm
useclock /Vm100 2

create table /Vm200
call /Vm200 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /Vm200 step_mode 3
addmsg /axon/c200 /Vm200 INPUT Vm
useclock /Vm200 2

create table /Vm300
call /Vm300 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /Vm300 step_mode 3
addmsg /axon/c300 /Vm300 INPUT Vm
useclock /Vm300 2

create table /Vm400
call /Vm400 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /Vm400 step_mode 3
addmsg /axon/c400 /Vm400 INPUT Vm
useclock /Vm400 2


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
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
openfile "axon0.plot" w
writefile "axon0.plot" "/newplot"
writefile "axon0.plot" "/plotname Vm(0)"
closefile "axon0.plot"
tab2file axon0.plot /Vm0 table

openfile "axon100.plot" w
writefile "axon100.plot" "/newplot"
writefile "axon100.plot" "/plotname Vm(100)"
closefile "axon100.plot"
tab2file axon100.plot /Vm100 table

openfile "axon200.plot" w
writefile "axon200.plot" "/newplot"
writefile "axon200.plot" "/plotname Vm(200)"
closefile "axon200.plot"
tab2file axon200.plot /Vm200 table

openfile "axon300.plot" w
writefile "axon300.plot" "/newplot"
writefile "axon300.plot" "/plotname Vm(300)"
closefile "axon300.plot"
tab2file axon300.plot /Vm300 table

openfile "axon400.plot" w
writefile "axon400.plot" "/newplot"
writefile "axon400.plot" "/plotname Vm(400)"
closefile "axon400.plot"
tab2file axon400.plot /Vm400 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots written to axon*.plot. Each plot is a trace of membrane potential at a  %
% different point along the axon.                                               %
% If you have gnuplot, run 'gnuplot axon.gnuplot' to view the graphs.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
