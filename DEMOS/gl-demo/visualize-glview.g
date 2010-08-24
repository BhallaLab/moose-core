// moose


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
First run the glcell client:
	glcellclient -p 9999 -c colormaps/rainbow2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

int USE_SOLVER = 1

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 2.5e-5
float IODT = 1e-4
float VIZDT = 1e-3
float SIMLENGTH = 5
float INJECT = 5.0e-10

include channels/bulbchan.g

ce /library
	make_LCa3_mit_usb
	make_Na_rat_smsnn
	make_KA_bsg_yka
	make_KM_bsg_yka
	make_K_mit_usb
	make_K2_mit_usb
	make_Na_mit_usb
	
	// MOOSE cannot deal with this channel, at this time.
	// make_Kca_mit_usb
	
	make_Ca_mit_conc
ce /

//=====================================
//  Create cells
//=====================================
readcell morphologies/mit.p /cell

//=====================================
//  Scaling diameters
//=====================================
ce /cell
float dia
float scale = 10

str c
foreach c ( { el /cell/##[TYPE=Compartment] } )
	dia = { getfield { c } dia }
	setfield { c } dia { scale * dia }
end

scale = 0.1 * 4
foreach c ( { el /cell/glom# } )
	dia = { getfield { c } dia }
	setfield { c } dia { scale * dia }
end

scale = 0.1 * 4
foreach c ( { arglist "/cell/primary_dend[1] /cell/primary_dend[2] /cell/primary_dend[3] /cell/primary_dend[4] /cell/primary_dend[5]" } )
	dia = { getfield { c } dia }
	setfield { c } dia { scale * dia }
end

setfield soma dia 1e-4
setfield soma len 0.0
ce ..

//=====================================
//  Vis object
//=====================================
create GLview gl0
setfield gl0 vizpath /cell/##[CLASS=Compartment]
setfield gl0 host localhost
setfield gl0 port 9999
setfield gl0 bgcolor 050050050
setfield gl0 value1 Vm
setfield gl0 value1min -0.1
setfield gl0 value1max 0.05
setfield gl0 morph_val 1
setfield gl0 color_val 1
setfield gl0 sync off
setfield gl0 grid off

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {IODT}
setclock 4 {VIZDT}

useclock /gl0 4

//=====================================
//  Stimulus
//=====================================
setfield /cell/soma inject {INJECT}

//=====================================
//  Solvers
//=====================================
if ( ! USE_SOLVER )
	setfield /cell method "ee"
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -t

// quit
