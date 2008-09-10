// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
10 input neurons (all single compartmental) synapse with 100% connectivity
with 10 output neurons. The commands 'createmap' and 'planarconnect' are used
to establish the network. Exponential euler method is utilized.
This script runs on MOOSE as well as GENESIS.
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
	setfield compt Rm 1e10 Ra 1e8 Cm 1e-10 initVm -0.065 Em -0.065 Vm 0
	addmsg compt compt/Na_hh_tchan VOLTAGE Vm
	addmsg compt/Na_hh_tchan compt CHANNEL Gk Ek
	addmsg compt compt/K_hh_tchan VOLTAGE Vm
	addmsg compt/K_hh_tchan compt CHANNEL Gk Ek
	
	if ( MOOSE )
		create Cell in_cell
		create Cell out_cell
	else
		create neutral in_cell
		create neutral out_cell
	end
	
	//
	//  Input cell
	//
	copy compt in_cell/in_compt
	create spikegen in_cell/in_compt/spike
	setfield in_cell/in_compt/spike thresh -0.03 abs_refract 0.004
	addmsg in_cell/in_compt in_cell/in_compt/spike INPUT Vm
	
	//
	//  Output cell
	//
	copy compt out_cell/out_compt
	create synchan out_cell/out_compt/glu
	setfield out_cell/out_compt/glu tau1 0.001 tau2 0.002 gmax 1e-8 Ek 0.0
	addmsg out_cell/out_compt out_cell/out_compt/glu VOLTAGE Vm
	addmsg out_cell/out_compt/glu out_cell/out_compt CHANNEL Gk Ek
ce /

//=====================================
//  Create synapse
//=====================================
createmap /library/in_cell in_array 1 10
createmap /library/out_cell out_array 1 10
planarconnect \
	/in_array/in_cell[]/in_compt/spike \
	/out_array/out_cell[]/out_compt/glu \
	-probability 1 \
	-sourcemask box -100 -100 100 100 \
	-destmask box -100 -100 100 100        // 100 % connectivity

//planarweight /in_array/in_cell[]/in_compt/spike /out_array/out_cell[]/out_compt/glu -fixed 1
//planardelay /in_array/in_cell[]/in_compt/spike /out_array/out_cell[]/out_compt/glu -fixed 0

for ( i = 0; i < 10; i = i + 1 )
	for ( j = 0; j < 10; j = j + 1 )
		setfield /out_array/out_cell[{i}]/out_compt/glu \
			synapse[{j}].weight { i + j } \
			synapse[{j}].delay { i * j * 1e-4 }
	end
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
createmap table /plots 1 10 -object

for ( i = 0; i < 10; i = i + 1 )
	call /plots/table[{i}] TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
	useclock /plots/table[{i}] 2
	setfield /plots/table[{i}] step_mode 3
	addmsg /out_array/out_cell[{i}]/out_compt /plots/table[{i}] INPUT Vm
end

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Stimulus
//=====================================
setfield /in_array/in_cell[]/in_compt inject {INJECT}
if ( 0 )
// if ( MOOSE )
	setfield /in_array/in_cell[] method "ee"
	setfield /out_array/out_cell[] method "ee"
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

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	for ( i = 0; i < N_INPUT; i = i + 1 )
		create hsolve /in_array/in_cell[{i}]/solve
		setfield /in_array/in_cell[{i}]/solve \
			path /in_array/in_cell[{i}]/in_compt \
			comptmode 1  \
			chanmode 3
		call /in_array/in_cell[{i}]/solve SETUP
		setmethod 11
	end
	
	for ( i = 0; i < N_OUTPUT; i = i + 1 )
		create hsolve /out_array/out_cell[{i}]/solve
		setfield /out_array/out_cell[{i}]/solve \
			path /out_array/out_cell[{i}]/out_compt \
			comptmode 1  \
			chanmode 3
		call /out_array/out_cell[{i}]/solve SETUP
		setmethod 11
	end
end

//=====================================
//  Simulation
//=====================================
reset
step { SIMLENGTH } -t


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
// Clear the file
openfile "test.plot" w
writefile "test.plot" " "
closefile "test.plot"

// Open and close file with every iteration, to flush tab2file output
str filename
for (i = 0; i < 10; i = i + 1)
	filename = "test" @ {i} @ ".plot"
//	filename = "test.plot"
	openfile {filename} a
	writefile {filename} "/newplot"
	writefile {filename} "/plotname Vm"
//	flushfile {filename}
	closefile {filename}
	openfile {filename} a
	
	tab2file {filename} /plots/table[{i}] table
	writefile {filename} " "
	writefile {filename} " "
	closefile {filename}
end 


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'test.plot' written: contains membrane potential traces of the o/p neurons
Compare against reference.plot obtained using GENESIS.
If you have gnuplot, run 'gnuplot network.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
