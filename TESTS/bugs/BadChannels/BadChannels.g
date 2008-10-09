// genesis
// moose

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(bug-id: 2155915)

Loads in a number of channels and runs them with a square voltage/calcium pulse,
plots out their conductances.

19 out of 25 channels work: see README for details.

Pick one channel from script to run. Run in Genesis to obtain reference plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

float RUNTIME = 0.1
float TOT_RUNTIME = RUNTIME * 3
float DT = 1e-5
float PLOTDT = 1e-4
float EREST_ACT = -0.065

float PULSE_VOLTAGE = 0.1
float PULSE_CALCIUM = 10.0

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}

include compatibility.g

include channel-definitions.g

ce /library
/////////////////////////////////
//	These channels don't work.
//	Pick one to run.
/////////////////////////////////
	make_Kca_mit_usb            // Doesn't work in MOOSE yet
//	make_Ca_hip_traub           // Doesn't work in MOOSE yet
//	make_K_hip_traub            // Doesn't work in MOOSE yet
//	make_Kca_hip_traub          // Doesn't work in MOOSE yet
//	make_Kc_hip_traub91         // Doesn't work in MOOSE yet
//	make_Ca_bsg_yka             // Doesn't work in MOOSE yet

/////////////////////////////////
//	These channels work well:
/////////////////////////////////
//	make_LCa3_mit_usb
//	make_Na_rat_smsnn
//	make_KA_bsg_yka
//	make_KM_bsg_yka
//	make_K_mit_usb
//	make_K2_mit_usb
//	make_Na_mit_usb
//	make_Ca_hip_traub91
//	make_Kahp_hip_traub91
//	make_Na_hip_traub91
//	make_Kdr_hip_traub91
//	make_Ka_hip_traub91
//	make_Na_hh_tchan
//	make_K_hh_tchan
//	make_Na_rat_smsnn
//	make_Na_bsg_yka
//	make_KA_bsg_yka
//	make_KM_bsg_yka
//	make_K_bsg_yka
ce /

echo "
================================================================================
  Running simulation
================================================================================
"

create compartment /compt

setfield /compt \
	Em {EREST_ACT} \
	Rm 1e9	\
	Cm 1	\
	Ra 1e9	\
	initVm	-0.065	\
	Vm	-0.065

create Ca_concen /compt/ca
setfield ^ \
	tau 0.01333 \
	B	17.4e12	\
	Ca_base	0.0 \
	Ca 0.0
		
create table /plot
call /plot TABCREATE {TOT_RUNTIME / PLOTDT} 0 {RUNTIME}
useclock /plot 2
setfield /plot step_mode 3

str chan
str chname
str temp
str filename
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end

foreach chan ( { el /library/#[CLASS=channel] } )
	copy {chan} /compt
	chname = { getpath {chan} -tail }
	echo {chname}

	addmsg /compt /compt/{chname} VOLTAGE Vm
	addmsg /compt/{chname} /compt CHANNEL Gk Ek
	addmsg /compt/{chname} /plot INPUT Gk

	// Hack to control conc, Many channels don't like this message
	// but GENESIS is bad about zeroing out the Ca input variable, so
	// we need to put this in for consistency.
	if (	{ strcmp {chname} "Kca_mit_usb" } && \
			{ strcmp {chname} "Ca_hip_traub" } && \
			{ strcmp {chname} "K_hip_traub" } && \
			{ strcmp {chname} "Kca_hip_traub" } && \
			{ strcmp {chname} "Kc_hip_traub91" } && \
			{ strcmp {chname} "Ca_bsg_yka" } )
		addmsg /compt/ca /compt/{chname} CONCEN Ca 
	end

	reset

	setfield /compt Vm -0.065 x 0
	setfield /compt/ca Ca_base 0.0
	step {RUNTIME} -t

	setfield /compt Vm {PULSE_VOLTAGE}	x 0.5
	setfield /compt/ca Ca_base {PULSE_CALCIUM}
	step {RUNTIME} -t

	setfield /compt Vm -0.065 x 0
	setfield /compt/ca Ca_base 0.0
	step {RUNTIME} -t

	filename = {chname} @ extension
	openfile {filename} w
	writefile {filename} "/newplot"
	writefile {filename} "/plotname "{chname}
	closefile {filename}

	tab2file {filename} /plot table

	delete /compt/{chname}
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

quit
