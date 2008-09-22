// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reads in a mitral cell model, with 8 channel types. Solver is not employed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g
int USE_SOLVER = 0

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 1e-5
float IODT = 1e-4
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
	
	// MOOSE cannot deal with this channel, at this time.
	// make_Kca_mit_usb
	
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
addmsg /mit/soma /somaplot INPUT Vm

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

useclock /#[TYPE=table] 2

//=====================================
//  Stimulus
//=====================================
setfield /mit/soma inject {INJECT}

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /mit/solve
		setfield /mit/solve \
			path /mit/##[TYPE=symcompartment],/mit/##[TYPE=compartment] \
			chanmode 1
		call /mit/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /mit method "ee"
	end
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -t


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

filename = "mitral" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /somaplot table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plot in 'mitral.*.plot'. Reference plot in 'mitral.genesis.plot'

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
