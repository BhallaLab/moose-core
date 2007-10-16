//moose
include kholodenko.g

reset
step {MAXTIME} -t
echo done round 1

reset
step {MAXTIME} -t
echo done round 2

setfield /kinetics/MAPK/MAPK CoInit 0.5
reset
step {MAXTIME} -t
echo done round 3, using different initial cond

showmsg /graphs/conc1/MAPK.Co

setfield /kinetics method ee
reset
step {MAXTIME} -t

echo done round 4, using ee method
showmsg /graphs/conc1/MAPK.Co

setfield /kinetics method rk2
reset
step {MAXTIME} -t

echo done round 5, using rk2 method

save
echo done all
