// This creates a model of 2 dendritic compartments (d1 and d2)
// connected to a soma. Injection current applied to d1. The 
// voltages will be printed to stdout (look in process() in 
// HSolve.mh. 
// Each row in output corresponds to a timestep.
// Each column in the output represents a compartment (first 
// line gives the compartment name).
reset
create neutral /cell
create compartment /cell/soma
create compartment /cell/d1
create compartment /cell/d2
setfield /cell/d1 Ra 1.1
setfield /cell/d1 Em -0.065
setfield /cell/d1 Inject 0.001
setfield /cell/d2 Ra 1.2
setfield /cell/d2 Em -0.065
setfield /cell/soma Em -0.065
addmsg /cell/d1/raxial /cell/soma/axial
addmsg /cell/d2/raxial /cell/soma/axial

setclock 0 0.001
setclock 1 0.001
setclock 4 0.001

create Neutral /graphs
create Plot /graphs/d1
create Plot /graphs/d2
create Plot /graphs/soma
addmsg /graphs/d1/trigPlot /cell/d1/Vm
addmsg /graphs/d2/trigPlot /cell/d2/Vm
addmsg /graphs/soma/trigPlot /cell/soma/Vm

create HSolve /solver
setfield /solver path "/cell/##"

useclock /cell/##[TYPE=Compartment] 0
useclock /graphs/##[TYPE=Plot] 1
useclock /solver 4

reset
step 10 -t
call /graphs/##[TYPE=Plot] printIn branch_t0.001.plot
quit
