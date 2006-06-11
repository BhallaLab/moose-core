//moose

create Interpol /reference
showfield /reference xdivs
showfield /reference dx
showfield /reference table[1]
setfield /reference xdivs 100
setfield /reference table[1] 1234
showfield /reference xdivs
showfield /reference dx
showfield /reference table[1]

echo on now to HHGate
create HHGate /test
showfield /test A.xdivs
showfield /test A.dx
showfield /test A.table[1]
setfield /test A.xdivs 100
setfield /test A.table[1] 1234
showfield /test A.xdivs
showfield /test A.dx
showfield /test A.table[1]
