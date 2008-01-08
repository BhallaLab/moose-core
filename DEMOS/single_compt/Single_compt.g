// genesis && moose

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single compartment representing space clamped squid giant axon.               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
float SIMDT = 2e-6
float PLOTDT = 1e-4
float RUNTIME = 0.02
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
create compartment /library/compartment
setfield /library/compartment initVm -0.065 Vm -0.065 Em -0.065

ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell soma.p /axon
// A concession to compatibility with MOOSE
if ( {exists /axon method } )
	setfield /axon method ee
end

create table /Vm0
call /Vm0 TABCREATE {1 + RUNTIME / PLOTDT} 0 {RUNTIME}
setfield /Vm0 step_mode 3
addmsg /axon/soma /Vm0 INPUT Vm

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {PLOTDT}

useclock /axon/##[TYPE=compartment],/axon/##[TYPE=tabchannel] 0
useclock /axon/# 1 init
useclock /Vm0 2

reset
setfield /axon/soma inject {INJECT}
step {RUNTIME + SIMDT} -t

openfile "test.plot" a
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"
tab2file test.plot /Vm0 table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot is stored in 'test.plot'                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
