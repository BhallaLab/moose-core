// moose
setfield /sli_shell isInteractive 0
int INSTANTX = 1
int INSTANTY = 2
int INSTANTZ = 4
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
setfield /cell/soma Inject 2e-10
setfield /sli_shell isInteractive 1
call /library/# reinitIn

create Plot /soma
addmsg /soma/trigPlot /cell/soma/Vm
create Plot /ca
addmsg /ca/trigPlot /cell/soma/Ca_conc/Ca
create Plot /K_AHP
addmsg /K_AHP/trigPlot /cell/soma/K_AHP/Gk
create Plot /K_C
addmsg /K_C/trigPlot /cell/soma/K_C/Gk


setclock 0 1e-5
setclock 1 1e-4
useclock /cell/##[TYPE=Compartment],/cell/##[TYPE=CaConc] 0
useclock /##[TYPE=Plot] 1

reset
step 0.1 -t
call /##[TYPE=Plot] printIn ca1.plot
quit
