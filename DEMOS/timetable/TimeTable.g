// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: TimeTable object connected to a single compartment via a SynChan.
       The TimeTable object reads spike-times from a text file, and emits
       spikes at the specified times.
Plots: - Vm of postsynaptic compartment
       - Gk of synaptic channel (embedded in compartment)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 50e-6
float SIMLENGTH = 60e-3

//=====================================
//  Create single-compartment cell with a synaptic channel
//=====================================
create compartment /cc
setfield /cc \
	Ra 1.0e10 \
	Rm 1.0e7 \
	Cm 1.0e-10 \
	initVm -0.07 \
	Em -0.07

ce /cc
	create synchan syn
	setfield ^ \
		Ek 0.0 \
		gmax 1.0e-9 \
		tau1 1.0e-3 \
		tau2 2.0e-3
	addmsg . syn VOLTAGE Vm
	addmsg syn . CHANNEL Gk Ek
ce /

//=====================================
//  Create TimeTable object
//=====================================
if ( GENESIS )
	create timetable /tt
	
	setfield /tt \
		method 4 \
		fname spikes.txt \
		act_val { 1.0 / SIMDT } \
		maxtime { SIMLENGTH } // Setting maxtime is necessary in Genesis

	call /tt TABFILL
	
	addmsg /tt /cc/syn ACTIVATION activation
else
	create TimeTable /tt
	
	// Method is 4 by default, but setting it anyway.
	setfield /tt \
		method 4 \
		filename spikes.txt
	
	addmsg /tt/event /cc/syn/synapse
end


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /data

create table /data/Vm
call /data/Vm TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm step_mode 3

create table /data/Gk
call /data/Gk TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Gk step_mode 3

//=====================================
//  Record from compartment and channel
//=====================================
addmsg /cc /data/Vm INPUT Vm
addmsg /cc/syn /data/Gk INPUT Ik


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

useclock /data/#[TYPE=table] 2

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
str outdir = "output/"
str filename
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

filename = {outdir} @ "Vm" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /data/Vm table

filename = {outdir} @ "Gk" @ {extension}
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Gk"
closefile {filename}
tab2file {filename} /data/Gk table


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.moose.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
