//moose

create compartment /c1
create compartment /c2

addmsg /c1 /c2 AXIAL Vm
addmsg /c2 /c1 RAXIAL Ra Vm

showmsg /c1

showmsg /c2
