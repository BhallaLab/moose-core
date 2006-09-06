//moose

include kholodenko.g
// include kkit_reac.g
// include kkit_enz.g
// include kkit_MMenz.g
reset

create Stoich /s
create KineticHub /khub
addmsg /s/hub /khub/hub
/*
addmsg /s/molSizesOut /khub/molSizesIn
addmsg /s/molConnectionsOut /khub/molConnectionsIn
addmsg /s/reacConnectionsOut /khub/reacConnectionsIn
*/

setfield /s path "/kinetics/##"

create ForwardEuler /fe
addmsg /fe/integrate /s/integrate

useclock /fe 4

reset

step {MAXTIME} -t

setfield /sli_shell isInteractive 1
echo {getfield /kinetics/MAPK/MAPK n}
setfield /kinetics/MAPK/MAPK n 0.4
echo {getfield /kinetics/MAPK/MAPK n}
step {MAXTIME} -t
setfield /sli_shell isInteractive 0
setfield /kinetics/MAPK/int1/2 k3 0.6
step {MAXTIME} -t
call /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] printIn khub.plot

/*

//moose

include kholodenko.g
// include kkit_reac.g
// include kkit_enz.g
// include kkit_MMenz.g

setfield /kinetics/MAPK/int2 mode 4 // Buffer it
setfield /kinetics/MAPK/int4 mode 4 // Buffer it

setfield /sli_shell isInteractive 1

reset
create Ksolve /ksolve
setfield /ksolve path "/kinetics/##"

echo Trying to getfield MAPK n
echo { getfield /kinetics/MAPK/MAPK n }

echo Trying to setfield MAPK to 1.2321
setfield /kinetics/MAPK/MAPK n 1.2321

echo Trying to getfield MAPK again
echo { getfield /kinetics/MAPK/MAPK n }

echo Trying to getfield MKKK
echo { getfield /kinetics/MAPK/MKKK n }
echo Trying to getfield MKK
echo { getfield /kinetics/MAPK/MKK n }

echo Trying to getfield int2
echo { getfield /kinetics/MAPK/int2 n }

echo Trying to setfield int4 to 234.5
setfield /kinetics/MAPK/int4 nInit 234.5

echo Trying to setfield Neg_feedback kf to 11.11
setfield /kinetics/MAPK/Neg_feedback kf 11.11

echo Trying to setfield enz int1/2 k3 to 3.33
setfield /kinetics/MAPK/int1/2 k3 3.33

echo Trying to setfield enz int3/6 Km to 5.55
setfield /kinetics/MAPK/int3/6 Km 5.55
*/
