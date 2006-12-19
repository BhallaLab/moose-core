//genesis

float SIMDT = 1
float CONTROLDT = 10
float PLOTDT = 10
setclock 0 {SIMDT}	0
setclock 1 {SIMDT}	1
setclock 2 { CONTROLDT }
setclock 3 { PLOTDT }
setclock 4 {SIMDT} 1

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

create neutral /kinetics
create neutral /graphs
create Molecule /kinetics/mol1
showfield /kinetics/mol1 *

create neutral /kinetics/foo
create Molecule /kinetics/foo/mol2
showfield /kinetics/foo/mol2 *

create Reaction /kinetics/foo/mol2/zod
showfield /kinetics/foo/mol2/zod *

create Plot /graphs/glug
showfield /graphs/glug *
