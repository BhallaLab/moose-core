//moose
// This simulation tests readcell formation of a compartmental model
// of an axon.

float SIMDT = 1e-5
float PLOTDT = 1e-4
float RUNTIME = 0.05
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
create compartment /library/compartment

ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell soma.p /axon

create table /Vm0
call /Vm0 TABCREATE 5000 0 {RUNTIME}
setfield /Vm0 step_mode 3
// addmsg /Vm0/inputRequest /axon/soma/Vm
addmsg /axon/soma /Vm0 INPUT Vm

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {PLOTDT}

useclock /axon/##[TYPE=compartment],/axon/##[TYPE=tabchannel] 0
useclock /axon/# 1 init
useclock /##[TYPE=table] 2

reset
setfield /axon/soma inject {INJECT}
step {RUNTIME} -t

tab2file axon0.plot /Vm0 table
// setfield /Vm0 print "axon0.plot"

tab2file Na_xa.plot /axon/soma/Na_mit_usb X_A
// setfield /axon/soma/Na_mit_usb/xGate/A print "Na_xa.plot"
// setfield /axon/soma/Na_mit_usb/xGate/B print "Na_xb.plot"
// setfield /axon/soma/Na_mit_usb/yGate/A print "Na_ya.plot"
// setfield /axon/soma/Na_mit_usb/yGate/B print "Na_yb.plot"

// setfield /axon/soma/K_mit_usb/xGate/A print "K_xa.plot"
// setfield /axon/soma/K_mit_usb/xGate/B print "K_xb.plot"
// setfield /axon/soma/K_mit_usb/yGate/A print "K_ya.plot"
// setfield /axon/soma/K_mit_usb/yGate/B print "K_yb.plot"
