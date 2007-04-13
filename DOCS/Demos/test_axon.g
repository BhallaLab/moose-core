//moose
// This simulation tests readcell formation of a compartmental model
// of an axon.

float SIMDT = 1e-5
float PLOTDT = 1e-4
float RUNTIME = 0.5
float INJECT = 1e-10

include bulbchan.g

create neutral /library

ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell axon.p /axon

create Table /Vm0
setfield /Vm0 stepmode 3
addmsg /Vm0/inputRequest /axon/soma

create Table /Vm100
setfield /Vm100 stepmode 3
addmsg /Vm100/inputRequest /axon/c100

create Table /Vm200
setfield /Vm200 stepmode 3
addmsg /Vm200/inputRequest /axon/c200

create Table /Vm300
setfield /Vm300 stepmode 3
addmsg /Vm300/inputRequest /axon/c300

create Table /Vm400
setfield /Vm400 stepmode 3
addmsg /Vm400/inputRequest /axon/c400

setclock 0 {SIMDT} 0
setclock 1 {SIMDT} 1
setclock 2 {PLOTDT} 0

useclock /squid/## 0
useclock /squid/# 1 init
useclock /##[TYPE=Table] 2

setfield /axon/soma inject {INJECT}
/*
echo foo
// step 0.040 -t
setfield /Vm0 print "axon0.plot"
setfield /Vm1 print "axon1.plot"
setfield /Vm2 print "axon2.plot"
setfield /Vm3 print "axon3.plot"
setfield /Vm4 print "axon4.plot"
*/
