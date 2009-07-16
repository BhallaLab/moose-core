// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: Traub's 1991 model for Hippocampal CA3 pyramidal cell.
Plots: Vm and [Ca++] from soma.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g
int USE_SOLVER = 0


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

include traub91proto.g
if ( 1 )
ce /library
	make_Na
	make_Ca
	make_K_DR
	make_K_AHP
	make_K_C
	make_K_A
	make_Ca_conc
ce /
end
//=====================================
//  Create cells
//=====================================
if ( 0 )
	readcell CA3.p /cell0
	readcell CA3.p /cell1@1
else
//	readcell CA3-passive.p /cell0
//	readcell CA3-passive.p /cell1@1
	readcell CA3-one-chan.p /cell0
	readcell CA3-one-chan.p /cell1@1
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /p0
create neutral /p1@1

create table /p0/Vm0
call /p0/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p0/Vm0 stepmode 3

create table /p0/Ca0
call /p0/Ca0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p0/Ca0 stepmode 3

create table /p0/Vm1
call /p0/Vm1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p0/Vm1 stepmode 3

create table /p0/Ca1
call /p0/Ca1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p0/Ca1 stepmode 3

create table /p1/Vm0@1
call /p1/Vm0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p1/Vm0 stepmode 3

create table /p1/Ca0@1
call /p1/Ca0 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p1/Ca0 stepmode 3

create table /p1/Vm1@1
call /p1/Vm1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p1/Vm1 stepmode 3

create table /p1/Ca1@1
call /p1/Ca1 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /p1/Ca1 stepmode 3

//=====================================
//  Record from compartment
//=====================================
addmsg /cell0/soma /p0/Vm0 INPUT Vm
// addmsg /cell0/soma/Ca_conc /p0/Ca0 INPUT Ca

addmsg /cell0/soma /p1/Vm0 INPUT Vm
// addmsg /cell0/soma/Ca_conc /p1/Ca0 INPUT Ca

// addmsg /cell1/soma /p0/Vm1 INPUT Vm
// addmsg /cell1/soma/Ca_conc /p0/Ca1 INPUT Ca

addmsg /cell1/soma /p1/Vm1 INPUT Vm
// addmsg /cell1/soma/Ca_conc /p1/Ca1 INPUT Ca


if ( 1 )
////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /cell0/soma inject {INJECT}
setfield /cell1/soma inject {INJECT}

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

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
		create hsolve /cell0/solve
		setfield /cell0/solve \
			path /cell0/##[TYPE=symcompartment],/cell0/##[TYPE=compartment] \
			chanmode 1
		call /cell0/solve SETUP
		setmethod 11
		
		create hsolve /cell1/solve
		setfield /cell1/solve \
			path /cell1/##[TYPE=symcompartment],/cell1/##[TYPE=compartment] \
			chanmode 1
		call /cell1/solve SETUP
		setmethod 11
	end
else
	if ( MOOSE )
		setfield /cell0 method "ee"
		setfield /cell1 method "ee"
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
	setfield /cell0/solve/integ CaAdvance 0
	setfield /cell1/solve/integ CaAdvance 0
end
//step {SIMLENGTH} -time


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

filename = "Vm00" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /p0/Vm0 table

filename = "Ca00" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Ca"
closefile {filename}
tab2file {filename} /p0/Ca0 table

filename = "Vm11" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /p1/Vm1 table

filename = "Ca11" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Ca"
closefile {filename}
tab2file {filename} /p1/Ca1 table


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
//quit
end
