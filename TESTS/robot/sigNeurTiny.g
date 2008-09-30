//genesis
// This version sets up the cell model, the signaling model and the
// adaptors.

// Create a library of prototype elements to be used by the cell reader
include proto16.g

create KinCompt /kinetics
create Molecule /kinetics/Ca_input
setfield /kinetics/Ca_input concInit 0.08

create Molecule /kinetics/K_A
setfield /kinetics/K_A concInit 1
// setfield /kinetics/K_A D 1

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
setfield /sig channelMap[K_A] K_A

showfield /sig *
le /sig

setfield /sig build foo
reset
reset

showclocks

showfield /sig/cell/soma/K_A Gbar
showfield /sig/kinetics/soma[1]/K_A n
showfield /sig/cell/soma/Ca_conc Ca
showfield /sig/kinetics/soma[1]/Ca_input n nInit
step 0.1 -t
showfield /sig/cell/soma/K_A Gbar
showfield /sig/kinetics/soma[1]/K_A n
setfield /sig/kinetics/soma[1]/K_A n 1e8 nInit 1e8
showfield /sig/cell/soma/Ca_conc Ca
showfield /sig/kinetics/soma[1]/Ca_input n nInit
step 0.1 -t
showfield /sig/cell/soma/K_A Gbar
showfield /sig/kinetics/soma[1]/K_A n
showfield /sig/cell/soma/Ca_conc Ca
showfield /sig/kinetics/soma[1]/Ca_input n nInit
