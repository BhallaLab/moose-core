// moose
setfield /sli_shell isInteractive 0

create neutral /library

create neutral /cable
readcell cable.p /cable

/*
setfield /cable/# Rm 2.65e9
setfield /cable/# Cm 1.13e-11
setfield /cable/# Ra 152.8e6
*/

setfield /sli_shell isInteractive 1

create Plot /soma
addmsg /soma/trigPlot /cable/soma/Vm
create Plot /dend20
addmsg /dend20/trigPlot /cable/dend20/Vm
setfield /cable/dend20 Inject 1e-7

setclock 0 1e-5
setclock 1 1e-3
useclock /cable/##[TYPE=Compartment],/cable/##[TYPE=CaConc] 0
useclock /##[TYPE=Plot] 1

reset
step 0.1 -t
call /##[TYPE=Plot] printIn cable2.plot
