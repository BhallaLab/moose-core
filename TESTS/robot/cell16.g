
//genesis

int hflag = 1    // use hsolve if hflag = 1
int chanmode = 1
int DO_X = 1

/* chanmodes 0 and 3 allow outgoing messages to non-hsolved elements.
   chanmode 3 is fastest.
*/

// All times are in sec
float dt = 50e-6                // simulation time step in sec
float PLOTDT = 0.5e-3
float SETTLE_TIME = 10.0		// Time to allow model to settle.
setclock  0  {dt}               // set the simulation clock
setclock  1  {PLOTDT}           // set the output clock
float injcurr = 2.5e-9		// default injection, Amps.
float frequency = 100.0		// default input frequency
float D = 50e-12			// Diffusion const for Ca. See
							// Allbritton, Meyer and Stryer, Science 92
float F = 96845.4				// Faraday const
float NUM_SPINES_PER_COMPT = 1.0	// Scaling for Ca conc and current

// Create a library of prototype elements to be used by the cell reader
include proto16.g

create neutral /library
pushe /library
//    make_cylind_symcompartment  /* makes "symcompartment" */
create symcompartment symcompartment
/* Assign some constants to override those used in traub91proto.g */
EREST_ACT = -0.06       // resting membrane potential (volts)
float ENA = 0.115 + EREST_ACT // 0.055  when EREST_ACT = -0.060
float EK = -0.015 + EREST_ACT // -0.075
float ECA = 0.140 + EREST_ACT // 0.080
float ACTIVATION_STRENGTH = 1.0;

make_Na
make_Ca
make_K_DR
make_K_AHP
make_K_C
make_K_A
make_Ca_conc
make_glu
make_NMDA
make_Ca_NMDA
make_NMDA_Ca_conc

pope

//===============================
//      Function Definitions
//===============================

function do_save_plots( file )
	str file

	if (DO_X)
		str plot = "/data/voltage/somaVm"
		int x = {getfield {plot} npts}
		echo "" > {file}

		str name
		str tail
		foreach name ( {el /data/#/#[TYPE=xplot]} )
			tail = { getpath {name} -tail }
			echo "/newplot" >> {file}
			echo "/plotname "{tail} >> {file}
			tab2file {file} {name} xpts -table2 ypts -nentries {x - 1}
		end
	else
		str plot = "/data/voltage/soma"
		int x = {getfield {plot} output } - 1
		echo "" > {file}
		foreach name ( {el /data/conc/tab_#[]} )
			tail = { getpath {name} -tail }
			echo "/newplot" >> {file}
			echo "/plotname "{tail} >> {file}
			tab2file {file} {name} table -nentries {x}
		end
		foreach name ( {el /data/voltage/#[]} )
			tail = { getpath {name} -tail }
			echo "/newplot" >> {file}_Vm
			echo "/plotname "{tail} >> {file}_Vm
			tab2file {file}_Vm {name} table -nentries {x}
		end
	end
end

// path is path of synapses to be activated
function setup_stim( path )
	str path
	str name

	create neutral /stim
	foreach name ( {el {path} } )
		addmsg /stim {name}/glu ACTIVATION x
		addmsg /stim {name}/NMDA ACTIVATION y
		addmsg /stim {name}/Ca_NMDA ACTIVATION z
	end
	//setfield {path}/NMDA/block CMg 0
end

// Frequency is stimulus frequency
// Time is duration to apply stimulus.
// The stimulus input connectivity is set up by setup_stim, above.
function do_run( freq, time, flag )
	float freq
	float time 
	int flag

	if ( freq <= ( 1.0 / time ) )
		step { time } -t
	else
		float inter_stim_time = 1.0 / freq
		float actx = ACTIVATION_STRENGTH / dt
		float acty = ACTIVATION_STRENGTH / dt
		float actz = ACTIVATION_STRENGTH / dt
		if ( ! ( flag & 1 ) )
			actx = 0;
		end
		if ( ! ( flag & 2 ) )
			acty = 0;
		end
		if ( ! ( flag & 4 ) )
			actz = 0;
		end
		int nstim = time * freq
		// echo nstim = {nstim}, time = {time}, freq = {freq}, actx = {actx}, flag = {flag}
		int i
	
		for ( i = 0 ; i < nstim; i = i + 1 )
			setfield /stim x {actx}
			setfield /stim y {acty}
			setfield /stim z {actz}
			step 1
			setfield /stim x 0
			setfield /stim y 0
			setfield /stim z 0
			step { inter_stim_time - dt } -t
		end
	end
end

//===============================
//    Graphics Functions
//===============================

function make_control
    create xform /control [10,0,250,170]
    create xlabel /control/label -hgeom 40 -bg cyan -label "CONTROL PANEL"
    create xbutton /control/RESET -wgeom 33%       -script reset
    create xbutton /control/RUN  -xgeom 0:RESET -ygeom 0:label -wgeom 33% \
         -script "do_run 10 1 7"
    create xbutton /control/QUIT -xgeom 0:RUN -ygeom 0:label -wgeom 34% \
        -script quit
    create xdialog /control/Injection -label "Injection (amperes)" \
		-value {injcurr}  -script "set_inject <widget>"
    create xdialog /control/stepsize -title "dt (sec)" -value {dt} \
                -script "change_stepsize <widget>"
    create xtoggle /control/overlay  \
           -script "overlaytoggle <widget>"
    setfield /control/overlay offlabel "Overlay OFF" onlabel "Overlay ON" state 0

    xshow /control
end

function make_real_graphs
	int i
    float vmin = -0.110
    float vmax = 0.05
    create xform /data [265,0,750,350]
    create xlabel /data/label -hgeom 10% -label "modified Traub CA1 Pyramidal Cell "
    create xgraph /data/voltage  -hgeom 90% -wgeom 50%  -title "Membrane Potential"
    setfield ^ XUnits sec YUnits Volts
    setfield ^ xmax {0.6} ymin {vmin} ymax {vmax}
    create xgraph /data/conc [0:last,0:label,50%,90%] -title "Ca Concentration"
    setfield ^ XUnits sec YUnits Arbitrary
    setfield ^ xmax {0.6} ymin 0 ymax 150
	addmsg /cell/soma /data/voltage PLOT Vm *somaVm *red
	addmsg /cell/soma/Ca_conc /data/conc PLOT Ca *somaconc *red

	for ( i = 4; i <= 6; i= i + 1 )
		addmsg /cell/spine_head_15_{i}/NMDA_Ca_conc /data/conc PLOT Ca *spineCa{i} *{i * 3}
		addmsg /cell/apical_15_{i}/Ca_conc /data/conc PLOT Ca *dendCa{i} *{ 15 + i * 3}
		addmsg /cell/spine_head_15_{i} /data/voltage PLOT Vm *spine{i} *{i * 3}
		addmsg /cell/apical_15_{i} /data/voltage PLOT Vm *dend{i} *{15 + i * 3}
	end
	setfield /data/#/#[TYPE=xplot] ysquish 0
	create xshape /data/voltage/stim -coords [0.16667,0.04,0][0.25,0.04,0] \
		-linewidth 4 -fg white
	create xshape /data/voltage/synstim -coords [0.5,0.04,0][1,0.04,0] \
		-linewidth 4 -fg red
    xshow /data
end

function make_xcell
    create xform /cellform [0,350,1000,400]
    create xdraw /cellform/draw [0,0,100%,100%]
    setfield /cellform/draw \
		xmin -0.0024 xmax -0.0001 ymin -0.4e-5 ymax 1.4e-5 \
        zmin -1e-3 zmax 1e-3 \
        transform z
    xshow /cellform
	create xshape /cellform/draw/Ca_spine -text "Ca in spine" \
		-coords [0,0,0] -tx -2120e-6 -ty 1.1e-5
	create xshape /cellform/draw/Ca_dendrite -text "Ca in dendrite" \
		-coords [0,0,0] -tx -2120e-6 -ty 6e-6
	create xshape /cellform/draw/voltage -text "Membrane potential" \
		-coords [0,0,0] -tx -2120e-6 -ty 1e-6
    echo creating xcell
    create xcell /cellform/draw/cell
    setfield /cellform/draw/cell colmin -0.1 colmax 0.1 \
        path /cell/##[TYPE=symcompartment] field Vm \
        diarange -50 script "echo <w> <v>"

    create xcell /cellform/draw/ca_cell
    setfield /cellform/draw/ca_cell colmin 0 colmax 100 \
        path /cell/##[TYPE=symcompartment] \
		field Ca fieldpath Ca_conc \
        diarange -50 script "echo <w> <v>" \
		ty 5e-6

    create xcell /cellform/draw/NMDA_ca_cell
    setfield /cellform/draw/NMDA_ca_cell colmin 0 colmax 100 \
        path /cell/##[TYPE=symcompartment] \
		field Ca fieldpath NMDA_Ca_conc \
        diarange -50 script "echo <w> <v>" \
		ty 1e-5

	/*
	create xvar /cellform/draw/input
	setfield /cellform/draw/input \
		varmode colorboxview \
		value_min[0] 0 value_max[0] {NUM_SPINES_PER_COMPT * 1.0e-9} \
		tx -0.0006 ty 1.2e-6
	addmsg /cell/spine_head_15_1/glu /cellform/draw/input VAL1 Gk
	setfield /cellform/draw/input/shape \
		coords [0,1e-6,0][-0.0005,0,0][0.0005,0,0][0,1e-6,0]
	setfield /cellform/draw/input/shape[1] \
		coords [0,2e-6,0][-0.0005,0,0][0.0005,0,0][0,2e-6,0]

	create xvar /cellform/draw/input2
	setfield /cellform/draw/input2 \
		varmode colorboxview \
		value_min[0] 0 value_max[0] {NUM_SPINES_PER_COMPT * 1.0e-9} \
		sizescale 1e-4 \
		tx -0.0006 ty 1.12e-5
	setfield /cellform/draw/input2/shape \
		coords [0,1e-6,0][-0.0005,0,0][0.0005,0,0][0,1e-6,0]
	setfield /cellform/draw/input2/shape[1] \
		coords [0,2e-6,0][-0.0005,0,0][0.0005,0,0][0,2e-6,0]

	addmsg /cell/spine_head_15_1/NMDA/block /cellform/draw/input2 VAL1 Gk
	*/
end

function set_inject(dialog)
    str dialog
    setfield /cell/soma inject {getfield {dialog} value}
end

function change_stepsize(dialog)
   str dialog
   dt =  {getfield {dialog} value}
   setclock 0 {dt}
   echo "Changing step size to "{dt}
end

// Use of the wildcard sets overlay field for all graphs
function overlaytoggle(widget)
    str widget
    setfield /##[TYPE=xgraph] overlay {getfield {widget} state}
end

function fix_table_value
	int i

	float Ca
	for ( i = 1; i <= 12; i = i + 1 )
		Ca = {getfield /cell/apical_15_{i}/Ca_conc Ca}
		setfield /data/conc/tab_15[{i}] table->table[0] {Ca}
		Ca = {getfield /cell/apical_16_{i}/Ca_conc Ca}
		setfield /data/conc/tab_16[{i}] table->table[0] {Ca}
	end
	i = 1
	Ca = {getfield /cell/apical_17_{i}/Ca_conc Ca}
	setfield /data/conc/tab_17[{i}] table->table[0] {Ca}
end

// This function adjusts electrical and other properties of all spines
// to match the NUM_SPINES_PER_COMPT scaling factor.
// Increases spine head XA, and membrane parms to match: Cm, Rm, and Ra
// Increases spine neck XA, and membrane parms to match: Cm, Rm, and Ra
// Scales B for spine head Ca: B = 5.2e-6 / shell volume
// Scales channel conductances in spine head
function scale_spine( i, j )
	int i	// sub_compt #
	int j	// compt #

	str head = "/cell/spine_head_" @ {j} @ "_" @ {i}
	str neck = "/cell/spine_neck_" @ {j} @ "_" @ {i}

	float Rm = {getfield {head} Rm}
	setfield {head} Rm { Rm / NUM_SPINES_PER_COMPT }

	float Ra = {getfield {head} Ra}
	setfield {head} Ra { Ra / NUM_SPINES_PER_COMPT }

	float Cm = {getfield {head} Cm}
	setfield {head} Cm { Cm * NUM_SPINES_PER_COMPT }

	Rm = {getfield {neck} Rm}
	setfield {neck} Rm { Rm / NUM_SPINES_PER_COMPT }

	Ra = {getfield {neck} Ra}
	setfield {neck} Ra { Ra / NUM_SPINES_PER_COMPT }

	Cm = {getfield {neck} Cm}
	setfield {neck} Cm { Cm * NUM_SPINES_PER_COMPT }

	float Bspine = {getfield {head}/NMDA_Ca_conc B}
	setfield {head}/NMDA_Ca_conc B {Bspine / NUM_SPINES_PER_COMPT}

	float gmax = {getfield {head}/glu gmax}
	setfield {head}/glu gmax {gmax * NUM_SPINES_PER_COMPT}

	gmax = {getfield {head}/NMDA gmax}
	setfield {head}/NMDA gmax {gmax * NUM_SPINES_PER_COMPT}

	gmax = {getfield {head}/Ca_NMDA gmax}
	setfield {head}/Ca_NMDA gmax {gmax * NUM_SPINES_PER_COMPT}

end

// Fill in the diffusion between Ca_concen in spine and in dend
// The volume ratio is 0.5^3 / ( 10 * 4 * 4 ) = 7.8125e-4
// The XA of the spine neck is 0.1 uM across, and 0.5 uM long.
// flux = gradient * D * XA / len 
// Here the conc scaling has to be built into the terms, so we get
// dC/dt = flux / vol
function fix_cell_msgs
	float PI = 3.1415926535
											// 0.1 micron diameter
	float NECK_XA = NUM_SPINES_PER_COMPT * 0.05e-6 * 0.05e-6 * PI 
	float VOL_RATIO = NUM_SPINES_PER_COMPT * 0.5 * 0.5 * 0.5 / ( 10 * 4 * 4 )
	float len = 0.5e-6
	float DEND_LEN = 10.0e-6
	float DEND_DIA = 4.0e-6
	float DEND_XA = DEND_DIA * DEND_DIA * PI / 4.0
	
	float SPINE_VOL = NUM_SPINES_PER_COMPT * 0.5e-6 * 0.5e-6 * PI * 0.25 * 0.5e-6
	float DEND_VOL = DEND_XA * DEND_LEN
	//float factor = 0.00058
	// float factor = F * 10000
	float factor = F * F
	
	float kf = factor * D * NECK_XA / ( len * F )
	float kb = factor * D * NECK_XA / ( len * F )
	float kaxial = factor * D * DEND_XA / ( DEND_LEN * F )
	float ka
	int i,j

	// echo kf = {kf}, kf/spine = {kf / SPINE_VOL}, kf/dend = {kf / DEND_VOL}
	str lastCa = "/cell/apical_10/Ca_conc"
	addmsg {lastCa} {lastCa} fI_Ca Ca { -kaxial }
	
	for ( j = 11; j <= 18; j = j + 1 )
		for ( i = 1; i <= 12; i= i + 1 )
			scale_spine {i} {j}

			float Bspine = {getfield /cell/spine_head_{j}_{i}/NMDA_Ca_conc B}
			float Bdend = {getfield /cell/apical_{j}_{i}/Ca_conc B}

			if (  ( i == 12 ) && ( j == 18 ) )
				ka = kaxial
			else
				ka = kaxial * 2.0
			end
		
			// Remove the CHANNEL message from NMDA receptor to compartment
			deletemsg /cell/spine_head_{j}_{i}/NMDA 1 -outgoing
			deletemsg /cell/spine_head_{j}_{i}/Ca_NMDA 1 -outgoing
		
			// removing Ca from spine head.
			addmsg /cell/spine_head_{j}_{i}/NMDA_Ca_conc /cell/spine_head_{j}_{i}/NMDA_Ca_conc fI_Ca Ca {-kf }
		
			// Adding Ca to dend
			addmsg /cell/spine_head_{j}_{i}/NMDA_Ca_conc /cell/apical_{j}_{i}/Ca_conc fI_Ca Ca {kf }
		
			// removing Ca from dend
			addmsg /cell/apical_{j}_{i}/Ca_conc /cell/apical_{j}_{i}/Ca_conc fI_Ca Ca { -kf -ka }
		
			// Adding Ca to spine_head
			addmsg /cell/apical_{j}_{i}/Ca_conc /cell/spine_head_{j}_{i}/NMDA_Ca_conc fI_Ca Ca {kf}


			// Adding Ca from current to last compartment
			addmsg /cell/apical_{j}_{i}/Ca_conc {lastCa} fI_Ca Ca {kaxial}
			// Adding Ca from last compartment to current
			addmsg {lastCa} /cell/apical_{j}_{i}/Ca_conc fI_Ca Ca {kaxial}
			lastCa = "/cell/apical_" @ j @ "_" @ i @ "/Ca_conc"
		
		end
	end
end

function make_table_graphs
	int i

	create neutral /data
	create neutral /data/voltage
	create neutral /data/conc
	for ( i = 1; i <= 12; i = i + 1 )
		create table /data/conc/tab_15[{i}]
		setfield /data/conc/tab_15[{i}] step_mode 3 // Fills up table
		call /data/conc/tab_15[{i}] TABCREATE 200000 0 2e5
		addmsg /cell/apical_15_{i}/Ca_conc /data/conc/tab_15[{i}] INPUT Ca
	end

	for ( i = 1; i <= 12; i = i + 1 )
		create table /data/conc/tab_16[{i}]
		setfield /data/conc/tab_16[{i}] step_mode 3 // Fills up table
		call /data/conc/tab_16[{i}] TABCREATE 200000 0 2e5
		addmsg /cell/apical_16_{i}/Ca_conc /data/conc/tab_16[{i}] INPUT Ca
	end
	create table /data/conc/tab_17[1]
	setfield /data/conc/tab_17[1] step_mode 3 // Fills up table
	call /data/conc/tab_17[1] TABCREATE 200000 0 2e5
	addmsg /cell/apical_17_1/Ca_conc /data/conc/tab_17[1] INPUT Ca


	create table /data/voltage/tab_15
	setfield /data/voltage/tab_15 step_mode 3 // Fills up table
	call /data/voltage/tab_15 TABCREATE 200000 0 2e5
	addmsg /cell/apical_15_1 /data/voltage/tab_15 INPUT Vm

	create table /data/voltage/tab_16
	setfield /data/voltage/tab_16 step_mode 3 // Fills up table
	call /data/voltage/tab_16 TABCREATE 200000 0 2e5
	addmsg /cell/apical_16_1 /data/voltage/tab_16 INPUT Vm

	create table /data/voltage/soma
	setfield /data/voltage/soma step_mode 3 // Fills up table
	call /data/voltage/soma TABCREATE 200000 0 2e5
	addmsg /cell/soma /data/voltage/soma INPUT Vm

	create table /data/conc/soma
	setfield /data/conc/soma step_mode 3 // Fills up table
	call /data/conc/soma TABCREATE 200000 0 2e5
	addmsg /cell/soma/Ca_conc /data/conc/soma INPUT Ca
end

//===============================
//         Main Script
//===============================
// Build the cell from a parameter file using the cell reader
readcell cell16.p /cell

fix_cell_msgs

setup_stim "/cell/spine_head_15_#,/cell/spine_head_16_#"

int i,j

if ( DO_X )
	// make the control panel
	make_control
	make_real_graphs
	
	/* comment out the two lines below to disable the cell display (faster)  */
	make_xcell // create and display the xcell
	xcolorscale hot
	useclock /data/## 1
	useclock /cellform/## 1
else
	make_table_graphs
	useclock /data/##[] 1
end

if (hflag)
    create hsolve /cell/solve
    setfield /cell/solve path "/cell/##[][TYPE=symcompartment]"
    setmethod 11
    setfield /cell/solve chanmode {chanmode}
    call /cell/solve SETUP
    reset
    // echo "Using hsolve"
end

reset

// Run it for 1 sec to get a baseline, then the pulse for 1 sec,
// then a settling time of 18 sec.

// function do_run( freq, time, flag )
do_run 0 0.1 0
setfield /cell/soma inject 0
setfield /cell/apical_19 inject -1e-9
// setclock 1 {SETTLE_TIME}
do_run 0 0.05 0
setfield /cell/apical_19 inject 0
do_run 0 0.15 0

// There is a little glitch here for entering the table value
// fix_table_value
setclock 1 {PLOTDT}
// input 100 Hz on all receptors
// setfield /cell/soma inject {injcurr}
// do_run 0 0.5 7
setfield /cell/soma inject 0
//do_run 10 0.5 7
do_run 100 0.3 7

// do_save_plots C_alt1.plot
// quit
