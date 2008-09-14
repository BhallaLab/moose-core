// moose
// genesis

int NCELL = $1
int SIZE = $2
float SIM_LENGTH = $3
int PLOT = $4

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 100e-6
float SIMLENGTH = {SIM_LENGTH}
float INJECT = 2.0e-10
float EREST_ACT = -0.060
float ENA = 0.115 + EREST_ACT // 0.055  when EREST_ACT = -0.060
float EK = -0.015 + EREST_ACT // -0.075
float ECA = 0.140 + EREST_ACT // 0.080

include traub91proto.g

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
readcell CA3.p /proto/CA3

// SHOULD USE CREATEMAP
int i
for ( i = 1; i <= NCELL; i = i + 1 )
	copy /proto/CA3 /cell{i}
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
if ( PLOT )
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
	addmsg /cell1/soma /plots/Vm INPUT Vm
	addmsg /cell1/soma/Ca_conc /plots/Ca INPUT Ca
end


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
for ( i = 1; i <= NCELL; i = i + 1 )
	setfield /cell{i}/soma inject {INJECT}
end

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

if ( PLOT )
	useclock /plots/Vm 2
	useclock /plots/Ca 2
end

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	for ( i = 1; i <= NCELL; i = i + 1 )
		create hsolve /cell{i}/solve
		setfield /cell{i}/solve \
			path /cell{i}/##[TYPE=symcompartment],/cell{i}/##[TYPE=compartment] \
			chanmode 3
		call /cell{i}/solve SETUP
		setmethod 11
	end
end

//=====================================
//  Simulation
//=====================================
reset
/*
if ( MOOSE )
	for ( i = 1; i <= NCELL; i = i + 1 )
		setfield /cell{i}/solve/integ CaAdvance 0
	end
end
*/
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
if ( PLOT )
	str filename
	str extension
	if ( MOOSE )
		extension = ".moose.plot"
	else
		extension = ".genesis.plot"
	end

	filename = "Vm" @ {extension}
	openfile {filename} w
	writefile {filename} "/newplot"
	writefile {filename} "/plotname Vm"
	closefile {filename}
	tab2file {filename} /plots/Vm table

	filename = "Ca" @ {extension}
	openfile {filename} w
	writefile {filename} "/newplot"
	writefile {filename} "/plotname Ca"
	closefile {filename}
	tab2file {filename} /plots/Ca table
end


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference plot is included. Present curve is in test.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
