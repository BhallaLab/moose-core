// moose
setfield /sli_shell isInteractive 0
include traub91proto.g


create neutral /library

ce ^

make_Ca
make_Ca_conc
make_K_AHP
make_K_C
make_Na
make_K_DR
make_K_A

create neutral /cell
readcell CA1.p /cell

setfield /sli_shell isInteractive 1


le /library
le /library/Ca
le /cell
// showfield /library/Ca/X *
call /library/# reinitIn
showfield /library/Ca/Y *

create Plot /soma
addmsg /soma/trigPlot /cell/soma/Vm


setclock 0 1e-5
setclock 1 1e-3
useclock /cell/##[TYPE=Compartment],/cell/##[TYPE=CaConc] 0
useclock /##[TYPE=Plot] 1


reset
step 0.1 -t
call /##[TYPE=Plot] printIn traubCA1.plot
//quit
