//moose
setclock 0 0.01 0
setclock 1 0.01 1
setclock 2 0.01 0

create Neutral /kinetics
ce /kinetics
create Molecule A
setfield A nInit 1.0
create Molecule B
create Molecule E
setfield E nInit 1.0
create Enzyme E/enz
setfield E/enz mode 0
ce /

create Plot PA
create Plot PB
create Plot PE
create Plot PCplx


addmsg /kinetics/A/reac /kinetics/E/enz/sub
addmsg /kinetics/E/enz/prdOut /kinetics/B/reacIn
addmsg /kinetics/E/reac /kinetics/E/enz/enz

addmsg /PA/trigPlot /kinetics/A/n
addmsg /PB/trigPlot /kinetics/B/n
addmsg /PE/trigPlot /kinetics/E/n
addmsg /PCplx/trigPlot /kinetics/E/enz/enz_cplx/n

// addmsg /sched/cj/ct0/ /kinetics/P/
useclock /kinetics/##[TYPE=Molecule] 0
useclock /kinetics/##[TYPE=Enzyme] 1
useclock /##[TYPE=Plot] 2

reset
step 100 -t

call /PA printIn test_enz.plot
call /PB printIn test_enz.plot
call /PE printIn test_enz.plot
call /PCplx printIn test_enz.plot
// quit
