//genesis
// This version sets up the cell model, the signaling model and the
// adaptors.

// Create a library of prototype elements to be used by the cell reader
float RUNTIME = 0.1
float CELLDT = 50.0e-6
float SIGDT = 1.0e-3
float INJECT = 20e-12
float DEFAULT_VOL = 1.257e-16

include proto16.g

create KinCompt /kinetics
setfield /kinetics volume {DEFAULT_VOL}
include fix79.g
copy /kinetics /library/soma
copy /kinetics /library/dend
copy /kinetics /library/spine

delete /kinetics

ce /library
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

// create Neutral dend
// create Molecule dend/Ca_pool
// setfield dend/Ca_pool volumeScale 6e5 CoInit 0.08 D 1e-12
ce /

//===============================
//         Main Script
//===============================
// Build the cell from a parameter file using the cell reader
readcell cell16.p /library/cell
setfield /library/cell method ee

create SigNeur /sig
setfield /sig cellProto /library/cell
setfield /sig somaProto /library/soma
setfield /sig dendProto /library/dend
setfield /sig spineProto /library/spine
setfield /sig lambda 5.0001e-6
setfield /sig Dscale 1e-12
setfield /sig calciumMap[Ca_conc] Ca_input
setfield /sig channelMap[MAPK*] K_A
setfield /sig sigDt {SIGDT}
setfield /sig cellDt {CELLDT}
setfield /sig calciumScale 0.1

showfield /sig *
le /sig

setfield /sig build foo

create neutral /plots
function make_plot( name, path, field )
	str name
	str path
	str field

	create table /plots/{name}
	setfield /plots/{name} xmin 0 xmax {RUNTIME} xdivs {RUNTIME / SIGDT}
	setfield /plots/{name} dx {SIGDT}
	setfield /plots/{name} step_mode 3
	useclock /plots/{name} 3
	addmsg {path}/{field} /plots/{name}/inputRequest
end

make_plot Vm /sig/cell/soma Vm
// make_plot Ca_cell /sig/cell/soma/Ca_conc Ca
make_plot KA_cell /sig/cell/soma/K_A Gbar
make_plot Ca_sig /sig/kinetics/soma[1]/Ca_input conc
make_plot MAPK_sig /sig/kinetics/soma[1]/MAPK* conc

setfield /library/cell/soma inject {INJECT}

reset
showclocks

function dump_plots
	str plot
	foreach plot ( {el /plots/#} )
		echo {plot}
		openfile "test.plot" a
		writefile "test.plot" "/newplot"
		writefile "test.plot" "/plotname "{plot}
		closefile "test.plot"
		tab2file "test.plot" {plot} table
	end
end

step {RUNTIME} -t

dump_plots
