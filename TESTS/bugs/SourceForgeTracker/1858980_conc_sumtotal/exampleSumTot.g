//moose

create pool /mol
setfield /mol vol 2 slave_enable 1 nInit 0 n 0
create pool /src
setfield /src vol 1 slave_enable 4 nInit 1234
reset
addmsg /src /mol SUMTOTAL n nInit

reset

step 2
showfield /src n
showfield /mol n
setfield /mol slave_enable 2 n 0
step 2
showfield /src n
showfield /mol n


