//moose
setclock 0 0.01 0
setclock 1 0.01 1
setclock 2 0.01 0

create Neutral /kinetics
ce /kinetics
create Molecule A
setfield A nInit 1.0
create Molecule B
create Reaction R
create Molecule C
create Reaction R2
ce /

create Plot PA
create Plot PB
create Plot PC


addmsg /kinetics/A/reac /kinetics/R/sub
addmsg /kinetics/B/reac /kinetics/R/prd
addmsg /kinetics/B/reac /kinetics/R2/sub
addmsg /kinetics/C/reac /kinetics/R2/prd

addmsg /PA/trigPlot /kinetics/A/n
addmsg /PB/trigPlot /kinetics/B/n
addmsg /PC/trigPlot /kinetics/C/n

// addmsg /sched/cj/ct0/ /kinetics/P/
useclock /kinetics/##[TYPE=Molecule] 0
useclock /kinetics/##[TYPE=Reaction] 1
useclock /##[TYPE=Plot] 2

reset
step 100 -t

call /PA printIn test_reacreac.plot
call /PB printIn test_reacreac.plot
call /PC printIn test_reacreac.plot
// quit
