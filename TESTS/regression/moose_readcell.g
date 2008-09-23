//genesis
//moose

// This test program works the same in GENESIS and MOOSE. Loads in a 
// cell model, runs it, saves output in test.plot, quits.

float RUNTIME = 0.05
float DT = 1e-5
float PLOTDT = 1e-4
float EREST_ACT = -0.065

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}

addalias setup_table2 setupgate
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau
addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau

function settab2const(gate, table, imin, imax, value)
    str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
		setfield {gate} {table}->table[{i}] {value}
	end
end

include bulbchan.g

create neutral /library

ce /library
make_LCa3_mit_usb
make_Na_rat_smsnn
make_KA_bsg_yka
make_KM_bsg_yka
make_K_mit_usb
make_K2_mit_usb
make_Na_mit_usb
// make_Kca_mit_usb
// MOOSE cannot deal with this channel, at this time.
make_Ca_mit_conc
create compartment compartment
ce /

readcell mit.p /mit

create table /somaplot
call /somaplot TABCREATE {RUNTIME / PLOTDT} 0 {RUNTIME}
useclock /somaplot 2

setfield /somaplot step_mode 3

addmsg /mit/soma /somaplot INPUT Vm

setfield /mit/soma inject 5.0e-10
setfield /mit method ee

reset
step {RUNTIME} -t

openfile "test.plot" w
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"

tab2file test.plot /somaplot table
quit
