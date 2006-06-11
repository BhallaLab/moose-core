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
create Enzyme E/enz
setfield E/enz mode 1
create Reaction R
setfield R kf 0.2
setfield R kb 0.1
ce /

create Plot PA
create Plot PB
create Plot PE
create Plot PCplx


addmsg /kinetics/A/reac /kinetics/E/enz/sub
addmsg /kinetics/E/reac /kinetics/E/enz/enz
addmsg /kinetics/E/enz/prdOut /kinetics/B/reacIn
addmsg /kinetics/B/reac /kinetics/R/sub
addmsg /kinetics/A/reac /kinetics/R/prd

addmsg /PA/trigPlot /kinetics/A/conc
addmsg /PB/trigPlot /kinetics/B/conc
addmsg /PE/trigPlot /kinetics/E/conc
//addmsg /PCplx/trigPlot /kinetics/E/enz/enz_cplx/conc

useclock /kinetics/##[TYPE=Molecule] 0
useclock /kinetics/##[TYPE=Enzyme],/kinetics/##[TYPE=Reaction] 1
useclock /##[TYPE=Plot] 2

reset
step 100 -t

call /PA printIn test_MMenzreac.plot
call /PB printIn test_MMenzreac.plot
call /PE printIn test_MMenzreac.plot
// call /PCplx printIn test_MMenzreac.plot
// quit
