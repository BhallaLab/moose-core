//genesis

create Average m0
create Average m1
create Average m2

setfield m0 baseline 1.0
setfield m1 baseline 10.0
setfield m2 baseline 0.1

addmsg m0/output m1/input
addmsg m0/output m2/input
addmsg m1/output m2/input
addmsg m2/output m0/input

setclock 0 1 0

reset

/// The unit tests actually execute 19 steps
/// MOOSE executes upto and including 18 seconds, starting from t=0,
// hence 18 steps here.
step 18 -t

echo {getfield m0 mean}
echo {getfield m1 mean}
echo {getfield m2 mean}
