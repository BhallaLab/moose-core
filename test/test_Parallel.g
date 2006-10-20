//moose

create Neutral /kinetics
create Molecule /kinetics/m
setfield /kinetics/m nInit 1234
setfield /kinetics/m n 4231.0987
create Molecule /kinetics/n
create PostMaster /p


// This is NOT a shared message.
addmsg /kinetics/m/nOut /p/destIn
// addmsg /sched/cj/ct0/process /p/process

// This is also not a shared message
addmsg /p/srcOut  /kinetics/n/sumTotalIn

useclock /kinetics/##[TYPE=Molecule],/p 0


// call /p reinitIn
reset

step 10 -t
// call /p processIn
