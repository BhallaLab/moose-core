//moose
// This simulation tests readcell formation of a compartmental model
// of an axon.

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo loads a 501 compartment model of a linear excitable neuron. Current %
% injection in the first compartment leads to spiking, and the plots show       %
% propagation of action potentials along the length of the axon.                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

float SIMDT = 2e-5
float PLOTDT = 1e-4
float RUNTIME = 0.1
float INJECT = 1e-9
float EREST_ACT = -0.065

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

if ( !{exists /library} )
	create neutral /library
end
ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell axon.p /axon

create Table /Vm0
setfield /Vm0 stepmode 3
addmsg /Vm0/inputRequest /axon/soma/Vm

create Table /Vm100
setfield /Vm100 stepmode 3
addmsg /Vm100/inputRequest /axon/c100/Vm

create Table /Vm200
setfield /Vm200 stepmode 3
addmsg /Vm200/inputRequest /axon/c200/Vm

create Table /Vm300
setfield /Vm300 stepmode 3
addmsg /Vm300/inputRequest /axon/c300/Vm

create Table /Vm400
setfield /Vm400 stepmode 3
addmsg /Vm400/inputRequest /axon/c400/Vm

setclock 0 {SIMDT} 0
setclock 1 {SIMDT} 1
setclock 2 {PLOTDT} 0

useclock /axon/##[TYPE=Compartment],/axon/##[TYPE=HHChannel] 0
useclock /axon/# 1 init
useclock /##[TYPE=Table] 2

reset
setfield /axon/soma inject {INJECT}
step {RUNTIME} -t
setfield /Vm0 print "axon0.plot"
setfield /Vm100 print "axon1.plot"
setfield /Vm200 print "axon2.plot"
setfield /Vm300 print "axon3.plot"
setfield /Vm400 print "axon4.plot"

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots written to axon*.plot. Each plot is a trace of membrane potential at a  %
% different point along the axon.                                               %
% If you have gnuplot, run 'gnuplot axon.gnuplot' to view the graphs.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

quit
