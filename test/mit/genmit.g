// genesis 2.2 - "Stand-alone" version of the traub91 tutorial with
// hsolvable tab2Dchannel for K_C current

int hflag = 0    // use hsolve if hflag = 1
int chanmode = 3

/* chanmodes 0 and 3 allow outgoing messages to non-hsolved elements.
   chanmode 3 is fastest.
*/

float tmax = 0.1               // simulation time in sec
float dt = 1e-5              // simulation time step in sec
setclock  0  {dt}               // set the simulation clock
setclock  1  1e-4
float injcurr = 0.5e-9		// default injection

// Create a library of prototype elements to be used by the cell reader

/* file for standard compartments */
// include compartments
/* file for 1991 Traub model channels */
int INSTANTX = 1
int INSTANTY = 2
int INSTANTZ = 4
float EREST_ACT = -0.060       // resting membrane potential (volts)
// float ENA = 0.115 + EREST_ACT // 0.055  when EREST_ACT = -0.060
// float EK = -0.015 + EREST_ACT // -0.075
// float ECA = 0.140 + EREST_ACT // 0.080

addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau

function settab2const(gate, table, imin, imax, value)
	str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
	    setfield {gate} {table}->table[{i}] {value}
	end
end

include bulbchan.g
// include moosebulbchan.g

create neutral /library
pushe /library
//    make_cylind_symcompartment  /* makes "symcompartment" */
create symcompartment symcompartment
create compartment compartment
/* Assign some constants to override those used in traub91proto.g */

    /* These are some standard channels used in .p files */
    // make_Na_squid_hh
    // make_K_squid_hh
    // make_Na_mit_hh
    // make_K_mit_hh

    // make_Na_mit_tchan
    // make_K_mit_tchan

    /* There are some synaptic channels for the mitral cell */
    // make_glu_mit_upi
    // make_GABA_mit_upi

    make_LCa3_mit_usb
    make_K_mit_usb
    make_KA_bsg_yka
    make_K2_mit_usb
    make_Na_mit_usb
    make_Ca_mit_conc
    make_Kca_mit_usb

pope

//===============================
//      Function Definitions
//===============================

function do_save_plots
	str file = "genmit.plot"

	echo "/newplot" > {file}
	echo "/plotname Vm_genmit" >> {file}
	str plot = "/data/voltage/volts"
	int x = {getfield {plot} npts}

	tab2file {file} {plot} xpts -table2 ypts -nentries {x - 1}
	plot = "/data/voltage/conc"
	echo "/newplot" >> {file}
	echo "/plotname Ca_genmit" >> {file}
	tab2file {file} {plot} xpts -table2 ypts -nentries {x - 1}
end

function step_tmax
    step {tmax} -time
	do_save_plots
end

//===============================
//    Graphics Functions
//===============================

function make_control
    create xform /control [10,50,250,170]
    create xlabel /control/label -hgeom 40 -bg cyan -label "CONTROL PANEL"
    create xbutton /control/RESET -wgeom 33%       -script reset
    create xbutton /control/RUN  -xgeom 0:RESET -ygeom 0:label -wgeom 33% \
         -script step_tmax
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
	disable /control
end

function make_Vmgraph
    float vmin = -0.100
    float vmax = 0.05
    create xform /data [265,50,350,350]
    create xlabel /data/label -hgeom 10% -label "Bhalla Mitral Cell "
    create xgraph /data/voltage  -hgeom 90%  -title "Membrane Potential"
    setfield ^ XUnits sec YUnits Volts
    setfield ^ xmax {tmax} ymin {vmin} ymax {vmax}
    xshow /data
	useclock /data/## 1
end

function make_xcell
    create xform /cellform [620,50,400,400]
    create xdraw /cellform/draw [0,0,100%,100%]
    setfield /cellform/draw xmin -0.003 xmax 0.001 ymin -3e-3 ymax 1e-3 \
        zmin -1e-3 zmax 1e-3 \
        transform z
    xshow /cellform
    echo creating xcell
    create xcell /cellform/draw/cell
    setfield /cellform/draw/cell colmin -0.1 colmax 0.1 \
        path /cell/##[TYPE=compartment] field Vm \
        script "echo <w> <v>"
	useclock /cellform/## 1
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

//===============================
//         Main Script
//===============================
// Build the cell from a parameter file using the cell reader
readcell mit.p /cell

setfield /cell/soma inject {injcurr}

// make the control panel
make_control

// make the graph to display soma Vm and pass messages to the graph
make_Vmgraph
addmsg /cell/soma /data/voltage PLOT Vm *volts *red
addmsg /cell/soma/Ca_mit_conc /data/voltage PLOT Ca *conc *blue

/* comment out the two lines below to disable the cell display (faster)  */
make_xcell // create and display the xcell
xcolorscale hot

if (hflag)
    create hsolve /cell/solve
    setfield /cell/solve path "/cell/##[][TYPE=compartment]"
    setmethod 11
    setfield /cell/solve chanmode {chanmode}
    call /cell/solve SETUP
    reset
    echo "Using hsolve"
end

//check
reset

step_tmax
// quit
