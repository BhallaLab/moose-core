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

useclock /kinetics/##[TYPE=Molecule],/kinetics/##[TYPE=Table],/##[TYPE=Compartment] 0
useclock /kinetics/##[TYPE=Reaction],/kinetics/##[TYPE=Enzyme],/##[TYPE=ConcChan] 1
useclock /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] 3
reset

create Neutral /kinetics
create Neutral /node1/kinetics
create Molecule /kinetics/m
setfield /kinetics/m nInit 1234
setfield /kinetics/m n 4231.0987

create Molecule /node1/kinetics/n
create Neutral /node1/cell
create ConcChan /node1/kinetics/cc

create Reaction /node1/kinetics/r

addmsg /kinetics/m/nOut /node1/kinetics/cc/nIn
addmsg /kinetics/m/reac /node1/kinetics/r/sub

step 1 -t
// call /p processIn
echo showfield /kinetics/m n 
showfield /kinetics/m n 
echo showfield /node1/kinetics/cc n 
showfield /node1/kinetics/cc n 
echo showfield /node1/kinetics/r *
showfield /node1/kinetics/r * 
