//moose
// This is a 2-node simulation to test messaging.

// Set up the instantaneous scheduling first of all.
float SIMDT = 1
float CONTROLDT = 10
float PLOTDT = 10
setclock 0 {SIMDT}	0
setclock 1 {SIMDT}	1
setclock 2 { CONTROLDT }
setclock 3 { PLOTDT }
setclock 4 {SIMDT} 2

useclock /kinetics/##[TYPE=Molecule],/kinetics/##[TYPE=Table] 0
useclock /kinetics/##[TYPE=Reaction],/kinetics/##[TYPE=Enzyme],/kinetics/##[TYPE=ConcChan] 1
useclock /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] 3
/*
addmsg /sched/cj/clock /sched/cj/ct0/clock
addmsg /sched/cj/clock /sched/cj/ct1/clock
addmsg /sched/cj/clock /sched/cj/ct2/clock
addmsg /sched/cj/clock /sched/cj/ct3/clock
addmsg /sched/cj/clock /sched/cj/ct4/clock
*/

create Neutral /kinetics
create Molecule /kinetics/m
setfield /kinetics/m nInit 1234
setfield /kinetics/m n 4231.0987

