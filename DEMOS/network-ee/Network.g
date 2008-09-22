// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
10 input neurons (all single compartmental) synapse with 100% connectivity
with 10 output neurons. The commands 'createmap' and 'planarconnect' are used
to establish the network. Exponential euler method is utilized.
This script runs on MOOSE as well as GENESIS.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 10e-6
float IODT = 100e-6
float SIMLENGTH = 0.15
float EREST_ACT = -0.065
int N_INPUT = 10
int N_OUTPUT = 10
int i
int j

include hh_tchan.g

ce /library
	make_Na_hh_tchan
	make_K_hh_tchan
	
	//
	//  Prototype compartment
	//
	create compartment compt
	copy Na_hh_tchan compt
	copy K_hh_tchan compt
	setfield compt Rm 1e8 Ra 1e8 Cm 1e-10 initVm -0.065 Em -0.065 Vm 0
	addmsg compt compt/Na_hh_tchan VOLTAGE Vm
	addmsg compt/Na_hh_tchan compt CHANNEL Gk Ek
	addmsg compt compt/K_hh_tchan VOLTAGE Vm
	addmsg compt/K_hh_tchan compt CHANNEL Gk Ek
	
	//
	//  Input cell
	//
	copy compt in_compt
	pushe in_compt
		create spikegen spike
		setfield spike thresh -0.04 abs_refract 0.001
		addmsg . spike INPUT Vm
	pope
	
	//
	//  Output cell
	//
	copy compt out_compt
	pushe out_compt
		create synchan glu
		setfield glu tau1 0.001 tau2 0.002 gmax 1e-8 Ek 0.0
		addmsg . glu VOLTAGE Vm
		addmsg glu . CHANNEL Gk Ek
	pope
ce /

//=====================================
//  Create synapse
//=====================================
createmap /library/in_compt /in_array 1 {N_INPUT}
createmap /library/out_compt /out_array 1 {N_OUTPUT}
planarconnect \
	/in_array/in_compt[]/spike \
	/out_array/out_compt[]/glu \
	-sourcemask box -100 -100 100 100 \
	-destmask box -100 -100 100 100 \
	-probability 1                       // 100 % connectivity

//planarweight /in_array/in_compt[]/spike /out_array/out_compt[]/glu -fixed 1
//planardelay /in_array/in_compt[]/spike /out_array/out_compt[]/glu -fixed 0

for ( i = 0; i < {N_OUTPUT}; i = i + 1 )
	for ( j = 0; j < {N_INPUT}; j = j + 1 )
		setfield /out_array/out_compt[{i}]/glu \
			synapse[{j}].weight { i + j } \
			synapse[{j}].delay { i * j * 1e-4 }
	end
end


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
createmap table /plots 1 {N_OUTPUT} -object
for ( i = 0; i < N_OUTPUT; i = i + 1 )
	call /plots/table[{i}] TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	setfield /plots/table[{i}] step_mode 3
	addmsg /out_array/out_compt[{i}] /plots/table[{i}] INPUT Vm
end


////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

useclock /plots/table[] 2

//=====================================
//  Simulation
//=====================================
reset

step {SIMLENGTH / 2} -t
setfield /in_array/in_compt Vm 0.0
step {SIMLENGTH / 2} -t


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

filename = "network" @ {extension}

// Clear file contents
openfile {filename} w
closefile {filename}

for (i = 0; i < N_OUTPUT; i = i + 1)
	openfile {filename} a
	writefile {filename} "/newplot"
	writefile {filename} "/plotname Vm["{i}"]"
	flushfile {filename}
	tab2file {filename} /plots/table[{i}] table	
	writefile {filename} " "
	
	// Force tab2file output to be flushed
	closefile {filename}
end


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'network.*.plot' written: contains membrane potential traces of the o/p 
neurons. Compare against network.genesis.plot obtained using GENESIS.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
