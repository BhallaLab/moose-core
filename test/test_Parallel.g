//moose

create Neutral /kinetics
create Molecule /kinetics/m
setfield /kinetics/m nInit 1234
setfield /kinetics/m n 4231.0987
create Molecule /kinetics/n
create PostMaster /p1
create PostMaster /p2
create Neutral /cell
create CaConc /cell/ca1
create CaConc /cell/ca2
create CaConc /cell/ca3
create HHChannel /cell/KA1
create HHChannel /cell/KA2
create HHChannel /cell/KA3

// This sets up the data transfer between postmasters:
setfield /p1 remoteNode 2
setfield /p2 remoteNode 1

setfield /p1 localNode 1
setfield /p2 localNode 2

// This is NOT a shared message.
addmsg /kinetics/m/nOut /p1/destIn
addmsg /cell/ca1/concOut /p1/destIn
addmsg /cell/ca2/concOut /p1/destIn
addmsg /cell/ca3/concOut /p1/destIn
// addmsg /sched/cj/ct0/process /p/process

// This is also not a shared message
addmsg /p2/srcOut  /kinetics/n/sumTotalIn
addmsg /p2/srcOut  /cell/KA1/concenIn
addmsg /p2/srcOut  /cell/KA2/concenIn
addmsg /p2/srcOut  /cell/KA3/concenIn

// setclock clockNum dt [stage]
setclock 0 1 0
setclock 1 1 1
setclock 2 1 2

useclock /kinetics/##[TYPE=Molecule],/p1,/p2 0
useclock /cell/##[TYPE=CaConc] 1
useclock /cell/##[TYPE=HHChannel] 2

// call /p reinitIn
reset
reset

step 10 -t
// call /p processIn
