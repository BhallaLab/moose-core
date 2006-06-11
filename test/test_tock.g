//moose

create Tock /t0
create Tock /t1
create Tock /t2
create Tock /t3

setclock 0 0.1 0
setclock 1 0.1 1
setclock 2 0.1 1
setclock 3 1 0

useclock /t0 tick 0
useclock /t1 tick 1
useclock /t2 tick 2
useclock /t3 tick 3

reset

step 2 -t

setclock 0 0.5 0
setclock 1 0.5
setclock 2 0.5

step 8 -t
