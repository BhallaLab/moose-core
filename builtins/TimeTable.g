//genesis

create TimeTable tt1
create TimeTable tt2

call /tt1 load testtider.txt 0
call /tt2 load testtider2.txt 0

create spikegen s1
create spikegen s2

addmsg /tt1/state /s1/Vm
addmsg /tt2/state /s2/Vm

setclock 0 1 0

reset

/// The unit tests actually execute 19 steps
/// MOOSE executes upto and including 18 seconds, starting from t=0,
// hence 18 steps here.
step 3 -t

