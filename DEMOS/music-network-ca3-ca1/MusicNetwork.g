// moose / music
// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script can be run in any of the following ways:

	genesis MusicNetwork.g full
	moose MusicNetwork.g full
	mpirun -np 2 music net-1.music
	mpirun -np 4 music net-2.music

In its full form, 2 sets of neurons are created: Input and Output. The i-th
input neuron synapses onto the i+3 and i-3 output neurons. All the cells are
single compartment cells, with HH channels that allow them to show spiking
activity. The input neurons are injected with different amounts of current.

The simulation can also be run in the form of 2 half networks, communicating
through the MUSIC library:

For running using MUSIC, use one of the folowing configuration files:
	net-1.music: 1 source process + 1 dest process
	net-2.music: 2 source processes + 2 dest processes

(Open the README file for further details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////

int NUM_NODES
if ( MOOSE )
	 NUM_NODES = { getfield /shell numNodes }
end

int USE_HSOLVE = 0

//=====================================
// Mode in which the script will be run.
//   FULL: Full network
//   SRC : Half network with input neurons generating spikes
//   DEST: Half network with output neurons consuming spikes
//=====================================
int FULL
int SRC
int DEST

//=====================================
// Flag to indicate whether the MUSIC port is connected. Applicable in SRC and
// DEST modes only.
//=====================================
int CONNECTED

//=====================================
//  Simulation parameters
//=====================================

float SIMDT = 10e-6
float IODT = 100e-6
float MUSICDT = 10e-6
float SIMLENGTH = 0.03

int N_INPUT = 10
int N_OUTPUT = 10

// Specify number of output cells to record from. Membrane potential will be
// recorded from output cells separated by a regular interval.
int N_PLOTS = 5

float INJECT_MIN = 1e-10
float INJECT_MAX = 1e-9

////////////////////////////////////////////////////////////////////////////////
// Parsing script arguments
////////////////////////////////////////////////////////////////////////////////

FULL = 0
SRC  = 0
DEST = 0

str mode = $1
if ( { strcmp {mode} "full" } == 0 )
	FULL = 1
elif ( { strcmp {mode} "src" } == 0 )
	SRC  = 1
elif ( { strcmp {mode} "dest" } == 0 )
	DEST = 1
else
	echo "Warning: Incorrect arguments"
	echo "Usage: <simulator> <script> [full|src|dest]"
	echo
	echo "Assuming 'full' mode and continuing..."
	
	FULL = 1
end

if ( SRC || DEST )
	if ( GENESIS )
		echo "Error: Can run only 'full' mode in GENESIS."
		echo "(Will now exit...)"
		quit
	end
	
	if ( ! {exists /music} )
		echo "Error: This MOOSE does not have MUSIC support."
		echo "Install a MOOSE package which supports the MUSIC library, and try"
		echo "again."
		echo "(Will now exit...)"
		quit
	end
end

////////////////////////////////////////////////////////////////////////////////
// Creating MUSIC sources / sinks
////////////////////////////////////////////////////////////////////////////////

CONNECTED = 0

//==============================================================================
//
//  Syntax for creating MUSIC ports:
//     call /music addPort [in|out] [event|continuous|message] <name>
//
//==============================================================================

if ( SRC )
	call /music addPort out event output
	CONNECTED = { getfield /music/output isConnected }
end

if ( DEST )
	call /music addPort in event input
	CONNECTED = { getfield /music/input isConnected }
end

if ( ( !FULL ) && ( !CONNECTED ) )
	echo "Warning: MUSIC port is not connected"
end

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////

int i
int j

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

ce /library
	//
	//  Input cell
	//
	readcell CA3.p CA3
	pushe CA3/soma
		create spikegen spike
		setfield spike thresh -0.04 abs_refract 0.001
		
		addmsg . spike INPUT Vm
	pope
	
	//
	//  Output cell
	//
	readcell CA1.p CA1
	pushe CA1/apical_14
		create synchan syn
		setfield syn tau1 0.001 tau2 0.002 gmax 1e-8 Ek 0.0
		
		addmsg . syn VOLTAGE Vm
		addmsg syn . CHANNEL Gk Ek
	pope
ce /
if ( 0 )
//=====================================
//  HSolve
//=====================================
if ( MOOSE )
	if ( ! USE_HSOLVE )
		setfield /library/CA1 method ee
		setfield /library/CA3 method ee
	end
end

//=====================================
//  Create cells
//=====================================

if ( FULL || SRC )
	create neutral /input
	
	for ( i = 0; i < N_INPUT; i = i + 1 )
		copy /library/CA3 /input/c{i}
	end
end

if ( FULL || DEST )
	if ( ! {exists /output} )
	// GENESIS already has a /output
		create neutral /output
	end
	
	for ( i = 0; i < N_OUTPUT; i = i + 1 )
		copy /library/CA1 /output/c{i}
	end
end

//=====================================
//  Add stimulus
//=====================================

if ( FULL || SRC )
	float inject = { INJECT_MIN }
	float di = { { INJECT_MAX - INJECT_MIN } / { N_INPUT - 1 } }
	
	for ( i = 0; i < N_INPUT; i = i + 1 )
		setfield /input/c{i}/soma inject {inject}
		
		inject = inject + di
	end
end

//=====================================
//  Create synapses
//=====================================

str input
str output
str output1
str output2
int d1
int d2

if ( FULL || CONNECTED )
	if ( SRC )
		for ( i = 0; i < N_INPUT; i = i + 1 )
			input  = "/input/c" @ { i } @ "/soma/spike"
			output = "/music/output/channel[" @ { i } @ "]"
			
			addmsg {input}/event {output}/synapse
		end
	else
		for ( i = 0; i < N_INPUT; i = i + 1 )
			d1 = i - 3
			d2 = i + 3
			
			if ( d1 < 0 )
				d1 = N_INPUT + d1
			end
			
			if ( d2 >= N_INPUT )
				d2 = d2 % N_INPUT
			end
			
			d1 = d1 % N_OUTPUT
			d2 = d2 % N_OUTPUT
			
			if ( DEST )
				input = "/music/input/channel[" @ { i } @ "]"
			else
				input = "/input/c" @ { i } @ "/soma/spike"
			end
			
			output1 = "/output/c" @ { d1 } @ "/apical_14/syn"
			output2 = "/output/c" @ { d2 } @ "/apical_14/syn"
			
			if ( FULL )
				addmsg {input} {output1} SPIKE
				addmsg {input} {output2} SPIKE
			else
				addmsg {input}/event {output1}/synapse
				addmsg {input}/event {output2}/synapse
			end
		end
	end
end

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
if ( ! SRC )
	str recordCell
	str recordCompt = "soma"
	int jump = { N_OUTPUT / N_PLOTS }
	
	createmap table /plots 1 {N_PLOTS} -object
	for ( i = 0; i < N_PLOTS; i = i + 1 )
		recordCell = "/output/c" @ { i * jump }
		
		call /plots/table[{i}] TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
		setfield /plots/table[{i}] step_mode 3
		addmsg {recordCell}/{recordCompt} /plots/table[{i}] INPUT Vm
	end
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
setclock 3 {MUSICDT}

// Can connect /music to any clock, except clock 0.
// This is because MUSIC ports are connected to clock 0, and it is necessary
// that the ports get initialized before /music. This is guaranteed only if
// /music is connected to a later clock.
useclock /music 3
useclock /plots/table[] 2

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -t


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
if ( ! SRC )
	str filename
	if ( DEST )
		filename = "music-" @ {NUM_NODES} @ ".plot"
	elif ( MOOSE )
		filename = "moose-" @ {NUM_NODES} @ ".plot"
	else
		filename = "genesis.plot"
	end
	
	filename = "output/" @ {filename}
	
	// Clear file contents
	openfile {filename} w
	closefile {filename}
	
	jump = { N_OUTPUT / N_PLOTS }
	for (i = 0; i < N_PLOTS; i = i + 1)
		openfile {filename} a
		writefile {filename} "/newplot"
		writefile {filename} "/plotname Cell("{ i * jump }")"
		flushfile {filename}
		tab2file {filename} /plots/table[{i}] table	
		writefile {filename} " "
		
		// Force tab2file output to be flushed
		closefile {filename}
	end
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
If you run this simulation using MOOSE, GENESIS and MOOSE/MUSIC, then the
following output plots will be generated:

	output/moose.plot
	output/genesis.plot
	output/music-1.plot
	output/music-2.plot

After that, if you have gnuplot, you can give the folowing command:

	gnuplot plot.gnuplot

which will plot the above graphs, and write them as images in the output folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
//quit
end
