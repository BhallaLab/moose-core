//moose

create KineticManager /m
create Molecule /m/Molecule[0]
create Molecule /m/Molecule[1]
setfield /m/Molecule[0] nInit 1
setfield /m/Molecule[1] nInit 0
create Reaction /m/Reaction[0]
create Reaction /m/Reaction[1]
addmsg /m/Molecule[0] /m/Reaction[1] SUBSTRATE n
addmsg /m/Reaction[1] /m/Molecule[0] REAC A B

addmsg /m/Molecule[1] /m/Reaction[1] PRODUCT n
addmsg /m/Reaction[1] /m/Molecule[1] REAC B A

addmsg /m/Molecule[0] /m/Reaction[0] SUBSTRATE n

setfield /m method rk5
reset
showfield /m/Molecule[] n
showfield /m/Reaction[] kf kb
step 10 -t
showfield /m/Molecule[] n
showfield /m/Reaction[] kf kb
