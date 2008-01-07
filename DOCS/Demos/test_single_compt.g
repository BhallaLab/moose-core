//moose
// This simulation tests readcell formation of a compartmental model
// of an axon.

float SIMDT = 1e-5
float PLOTDT = 1e-4
float RUNTIME = 0.5
float INJECT = 1e-9

// settab2const sets a range of entries in a tabgate table to a constant
function settab2const(gate, table, imin, imax, value)
	str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
		setfield {gate} {table}->table[{i}] {value} 
	end
end

addalias setup_table2 setupgate
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau
addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau

include bulbchan.g

create neutral /library

ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell soma.p /axon

create Table /Vm0
setfield /Vm0 stepmode 3
addmsg /Vm0/inputRequest /axon/soma/Vm

setclock 0 {SIMDT} 0
setclock 1 {SIMDT} 1
setclock 2 {PLOTDT} 0

useclock /axon/##[TYPE=Compartment],/axon/##[TYPE=HHChannel] 0
useclock /axon/# 1 init
useclock /##[TYPE=Table] 2

reset
setfield /axon/soma inject {INJECT}
step 0.050 -t
setfield /Vm0 print "axon0.plot"

setfield /library/Na_mit_usb/xGate/A print "Na_xa.plot"
setfield /library/Na_mit_usb/xGate/B print "Na_xb.plot"
setfield /library/Na_mit_usb/yGate/A print "Na_ya.plot"
setfield /library/Na_mit_usb/yGate/B print "Na_yb.plot"

setfield /library/K_mit_usb/xGate/A print "K_xa.plot"
setfield /library/K_mit_usb/xGate/B print "K_xb.plot"
setfield /library/K_mit_usb/yGate/A print "K_ya.plot"
setfield /library/K_mit_usb/yGate/B print "K_yb.plot"
