// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reads in a mitral cell model, with 8 channel types. Solver is not employed.   %
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
float INJECT = 5.0e-10
float EREST_ACT = -0.065

include bulbchan.g

ce /library
	make_LCa3_mit_usb
	make_Na_rat_smsnn
	make_KA_bsg_yka
	make_KM_bsg_yka
	make_K_mit_usb
	make_K2_mit_usb
	make_Na_mit_usb
	// make_Kca_mit_usb
	// MOOSE cannot deal with this channel, at this time.
	make_Ca_mit_conc
ce /

//=====================================
//  Create cells
//=====================================
readcell mit.p /mit


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create table /somaplot
call /somaplot TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /somaplot step_mode 3
useclock /somaplot 2

//=====================================
//  Record from compartment
//=====================================
addmsg /mit/soma /somaplot INPUT Vm


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
setfield /mit/soma inject {INJECT}

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /mit/solve
	setfield /mit/solve \
		path /mit/##[TYPE=symcompartment],/mit/##[TYPE=compartment] \
		comptmode 1  \
		chanmode 3
	call /mit/solve SETUP
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

tab2file test.plot /somaplot table
echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot in 'test.plot'. Reference plot in 'reference.plot'                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
