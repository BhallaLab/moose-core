// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: 2 PulseGens (running in free run mode) connected to 2 AscFile objects.
       Both AscFiles record both PulseGens, but the columns are swapped.

Type "help AscFile -full" and "help PulseGen -full" to see documentation on
these objects.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 50e-6
float SIMLENGTH = 100e-3

//=====================================
//  Create PulseGen objects
//=====================================
create pulsegen /p1
setfield /p1 \
	baselevel 0.0 \
	level1 1.0 \
	width1 10e-3 \
	delay1 10e-3 \
	trig_mode 0		/* free run */

create pulsegen /p2
setfield /p2 \
	baselevel 4.0 \
	level1 2.0 \
	width1 2e-3 \
	delay1 1e-3 \
	level2 5.0 \
	width2 5e-3 \
	delay2 5e-3 \
	trig_mode 0		/* free run */


//=====================================
//  Create AscFile objects
//=====================================
str outdir = "output"
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

create asc_file /a1
setfield /a1 \
	filename {outdir}"/a1"{extension} \
	append 0

create asc_file /a2
setfield /a2 \
	filename {outdir}"/a2"{extension} \
	append 0

//=====================================
//  Connect PulseGens with AscFiles
//=====================================
if ( GENESIS )
	addmsg /p1 /a1 SAVE output
	addmsg /p2 /a1 SAVE output
	
	addmsg /p2 /a2 SAVE output
	addmsg /p1 /a2 SAVE output
else
	addmsg /p1/output /a1/save
	addmsg /p2/output /a1/save
	
	addmsg /p2/output /a2/save
	addmsg /p1/output /a2/save
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

useclock /#[TYPE=asc_file] 2

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time

echo "
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Hack: Calling the "close" function for AscFile:s explicitly to flush the file.
Should not be required, really. Currently a bug in MOOSE does not finalize
objects at exit.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"
call /a1 close
call /a2 close

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.moose.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
