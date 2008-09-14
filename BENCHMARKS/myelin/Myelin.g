// moose
// genesis

int NCELL = $1
int SIZE = $2
float SIM_LENGTH = $3
int PLOT = $4

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
float SIMLENGTH = {SIM_LENGTH}
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
readcell myelin2.p /proto/axon

// SHOULD USE CREATEMAP
int i
for ( i = 1; i <= NCELL; i = i + 1 )
	copy /proto/axon /axon{i}
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
if ( PLOT == 1 )
	create neutral /plot

	create table /plot/Vm0
	call /plot/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield /plot/Vm0 step_mode 3
	addmsg /axon1/soma /plot/Vm0 INPUT Vm

	create table /plot/Vm1
	call /plot/Vm1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield /plot/Vm1 step_mode 3
	addmsg /axon1/n99/i20 /plot/Vm1 INPUT Vm
end

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

if ( PLOT == 1 )
	useclock /plot/Vm0 2
	useclock /plot/Vm1 2
end

//=====================================
//  Stimulus
//=====================================
for ( i = 1; i <= NCELL; i = i + 1 )
	setfield /axon{i}/soma inject {INJECT}
end

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	for ( i = 1; i <= NCELL; i = i + 1 )
		create hsolve /axon{i}/solve
		setfield /axon{i}/solve \
			path /axon{i}/##[TYPE=symcompartment],/axon/##[TYPE=compartment] \
			chanmode 3
		call /axon{i}/solve SETUP
		setmethod 11
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
if ( PLOT == 1 )
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
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
