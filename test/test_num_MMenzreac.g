//moose
setclock 0 0.01 0
setclock 1 0.01 1
setclock 2 0.01 0

create Neutral /kinetics
ce /kinetics
create Molecule A
setfield A nInit 1.0
create Molecule B
// create Molecule C
create Molecule E
setfield E nInit 1.0
create Enzyme E/1
setfield E/1 mode 1
create Reaction R
setfield R kf 0.2
setfield R kb 0.1
ce /

create Plot PA
create Plot PB
create Plot PE
create Plot PCplx


addmsg /kinetics/A/reac /kinetics/E/1/sub
addmsg /kinetics/E/reac /kinetics/E/1/enz
addmsg /kinetics/E/1/prdOut /kinetics/B/reacIn
addmsg /kinetics/B/reac /kinetics/R/sub
addmsg /kinetics/A/reac /kinetics/R/prd

addmsg /PA/trigPlot /kinetics/A/conc
addmsg /PB/trigPlot /kinetics/B/conc
addmsg /PE/trigPlot /kinetics/E/conc
//addmsg /PCplx/trigPlot /kinetics/E/1/enz_cplx/conc

useclock /kinetics/##[TYPE=Molecule] 0
useclock /kinetics/##[TYPE=Enzyme],/kinetics/##[TYPE=Reaction] 1
useclock /##[TYPE=Plot] 2

reset
step 100 -t

call /PA printIn test_numMMenzreac.plot
call /PB printIn test_numMMenzreac.plot
call /PE printIn test_numMMenzreac.plot
// call /PCplx printIn test_MMenzreac.plot
// quit
