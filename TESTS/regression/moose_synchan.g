//genesis
//moose
//////////////////////////////////////////////////////////////////////
// This scripts tests the working of synapses connecting spike 
// objects to synchan objects.
// Works both with GENESIS and MOOSE.
//////////////////////////////////////////////////////////////////////

float DT = 10e-6
float PLOTDT = 100e-6
float RUNTIME = 0.1

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}

create compartment compt
setfield compt Rm 1e9
setfield compt Cm 1e-11
setfield compt Em -0.065
setfield compt initVm -0.06
copy /compt /incompt

create synchan compt/syn
setfield ^ Ek 0 gmax 1e-9 tau1 1e-3 tau2 2e-3

create spikegen /incompt/spike
setfield ^ thresh -0.04 abs_refract 0.01
addmsg /incompt /incompt/spike INPUT Vm

addmsg compt compt/syn VOLTAGE Vm
addmsg compt/syn compt CHANNEL Gk Ek
addmsg /incompt/spike /compt/syn SPIKE

create table /plot
call /plot TABCREATE {RUNTIME / PLOTDT} 0 {RUNTIME}
useclock /plot 2
setfield /plot step_mode 3

addmsg /compt/syn /plot INPUT Gk

reset
showmsg /compt

step {RUNTIME/2} -t
setfield /incompt Vm 0.0
step {RUNTIME/2} -t


openfile "test.plot" w
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Gk"
closefile "test.plot"

tab2file "test.plot" /plot table
quit
