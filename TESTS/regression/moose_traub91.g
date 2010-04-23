// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: Traub's 1991 model for Hippocampal CA3 pyramidal cell.
Plots: Vm and [Ca++] from soma.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include traub91/compatibility.g
int USE_SOLVER = 1


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 10e-6
float IODT = 100e-6
float SIMLENGTH = 0.10
float INJECT = 2.0e-10
float EREST_ACT = -0.060
float ENA = 0.115 + EREST_ACT // 0.055  when EREST_ACT = -0.060
float EK = -0.015 + EREST_ACT // -0.075
float ECA = 0.140 + EREST_ACT // 0.080

include traub91/traub91proto.g

ce /library
	make_Na
	make_Ca
	make_K_DR
	make_K_AHP
	make_K_C
	make_K_A
	make_Ca_conc
ce /

//=====================================
//  Create cells
//=====================================
readcell traub91/CA3.p /CA3


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plots

create table /plots/Vm
call /plots/Vm TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Vm step_mode 3

create table /plots/Ca
call /plots/Ca TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plots/Ca step_mode 3

//=====================================
//  Record from compartment
//=====================================
addmsg /CA3/soma /plots/Vm INPUT Vm
addmsg /CA3/soma/Ca_conc /plots/Ca INPUT Ca


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /CA3/soma inject {INJECT}

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {IODT}

useclock /plots/#[TYPE=table] 3

//=====================================
//  Solvers
//=====================================
// In Genesis, an hsolve object needs to be created.
//
// In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
// switching to the Exponential Euler method.

if ( USE_SOLVER )
	if ( GENESIS )
		create hsolve /CA3/solve
		setfield /CA3/solve \
			path /CA3/##[TYPE=symcompartment],/CA3/##[TYPE=compartment] \
			chanmode 1
		call /CA3/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /CA3 method "ee"
	end
end

//=====================================
//  Simulation
//=====================================
reset

//
// Genesis integrates the calcium current (into the calcium pool) in a slightly
// different way from Moose. While the integration in Moose is sligthly more
// accurate, here we force Moose to imitate the Genesis method, to get a better
// match.
//
if ( MOOSE && USE_SOLVER )
	setfield /CA3/solve/integ CaAdvance 0
end
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

filename = "test.plot"
//filename = "Vm" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /plots/Vm table

//filename = "Ca" @ {extension}
// Should compare this using relative error
/*
openfile {filename} a
writefile {filename} " "
writefile {filename} "/newplot"
writefile {filename} "/plotname Ca"
closefile {filename}
tab2file {filename} /plots/Ca table
*/

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
