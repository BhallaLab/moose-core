//genesis
// Create a library of prototype elements to be used by the cell reader
include proto16.g

include acc79.g
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

showfield /sig *
le /sig

setfield /sig build foo
reset
reset

