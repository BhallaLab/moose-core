// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Much like the axon demo, a linear excitable cell is created using readcell.   %
% Integration is done using the Hines' method.                                  %
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
float SIMLENGTH = 0.25
float INJECT = 1e-10
float EREST_ACT = -0.065

include chan.g

ce /library
	make_Na_mit_usb
	make_K_mit_usb
ce /

//=====================================
//  Create cells
//=====================================
readcell myelin2.p /axon


////////////////////////////////////////////////////////////////////////////////
// MODULATION
////////////////////////////////////////////////////////////////////////////////
float Gbar = 1.5e-2

// Modulation table
create table /modulate
call /modulate TABCREATE 100 0 {SIMLENGTH}
setfield /modulate step_mode 2
setfield /modulate stepsize 0
if ( MOOSE )
	addmsg /modulate/outputSrc /axon/n21/Na_mit_usb/Gbar
	addmsg /modulate/outputSrc /axon/n24/Na_mit_usb/Gbar
end

int i
for ( i = 0; i < 60; i = i + 1 )
	setfield /modulate table->table[{i}] {Gbar}
end

useclock /modulate 0

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plot

create table /plot/Vm0
call /plot/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Vm0 step_mode 3
addmsg /axon/soma /plot/Vm0 INPUT Vm
useclock /plot/Vm0 2

create table /plot/Vm1
call /plot/Vm1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Vm1 step_mode 3
addmsg /axon/n99/i20 /plot/Vm1 INPUT Vm
useclock /plot/Vm1 2


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
		chanmode 1
	call /axon/solve SETUP
	setmethod 11
end

//=====================================
//  Simulation
//=====================================
reset
if ( GENESIS )
	setfield /axon/n21/Na_mit_usb Gbar {Gbar}
	setfield /axon/n24/Na_mit_usb Gbar {Gbar}
	step {0.6 * SIMLENGTH} -time
	setfield /axon/n21/Na_mit_usb Gbar 0.0
	setfield /axon/n24/Na_mit_usb Gbar 0.0
	step {0.4 * SIMLENGTH} -time
else
	step {SIMLENGTH} -time
end


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
openfile "axon.0.plot" w
writefile "axon.0.plot" "/newplot"
writefile "axon.0.plot" "/plotname Vm(0)"
closefile "axon.0.plot"
tab2file axon.0.plot /plot/Vm0 table

openfile "axon.x.plot" w
writefile "axon.x.plot" "/newplot"
writefile "axon.x.plot" "/plotname Vm(100)"
closefile "axon.x.plot"
tab2file axon.x.plot /plot/Vm1 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots written to axon.*.plot.                                                   %
% If you have gnuplot, run 'gnuplot myelin.gnuplot' to view the graphs.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
