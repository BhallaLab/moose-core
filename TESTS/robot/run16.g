//genesis
// Create a library of prototype elements to be used by the cell reader
float RUNTIME = 0.04
float DT = 1e-5
float PLOTDT = 1e-4

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}

include proto16.g

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

create Molecule Ca_pool
setfield Ca_pool volumeScale 6e5 CoInit 0.08
ce /

//===============================
//         Main Script
//===============================
// Build the cell from a parameter file using the cell reader
readcell cell16.p /cell

create table /somaplot
call /somaplot TABCREATE {RUNTIME / PLOTDT} 0 {RUNTIME}
useclock /somaplot 2
setfield /somaplot step_mode 3

addmsg /cell/soma /somaplot INPUT Vm

setfield /cell/soma inject 5.0e-10
// setfield /cell method ee

echo starting

reset
reset
step {RUNTIME} -t

openfile "test.plot" a
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"

tab2file test.plot /somaplot table
echo done

// create SigNeur /sig
// setfield /sig cellProto /cell
// setfield /sig dendProto /library/Ca

// showfield /sig *

// setfield /sig build foo
