include compatibility.g
include loadMoc.g

float SIMDT = 1e-5
float PLOTDT = 1e-5
float PULSETIME = 5e-3
float RUNTIME = 3 * {PULSETIME}
float EM = -0.07
float RM = 424.4e3
float RA = 7639.44e3
float CM = 0.007854e-6

str sim
if ( GENESIS )
	sim = "genesis"
else
	sim = "moose"
end

create compartment cc
setfield /cc Rm {RM}
setfield /cc Ra {RA}
setfield /cc Cm {CM}
setfield /cc Em {EM}
ce cc
	make_Moczyd_KC
	setfield Moczyd_KC Gbar 100e-3
	
	addmsg . Moczyd_KC VOLTAGE Vm
	addmsg Moczyd_KC . CHANNEL Gk Ek
	
	create Ca_concen Ca_conc

	if ( GENESIS )
		addmsg Ca_conc Moczyd_KC CONCEN1 Ca
	else
		addmsg Ca_conc Moczyd_KC CONCEN Ca
	end
ce ..

create table /plot1
call /plot1 TABCREATE {RUNTIME / PLOTDT} 0 1
setfield /plot1 step_mode 3
addmsg /cc /plot1 INPUT Vm

create table /plot2
call /plot2 TABCREATE {RUNTIME / PLOTDT} 0 1
setfield /plot2 step_mode 3
addmsg /cc/Moczyd_KC /plot2 INPUT Gk

setclock 0 {SIMDT}
setclock 1 {PLOTDT}

useclock /cc,/cc/# 0
useclock /plot1 1
useclock /plot2 1

setfield /cc initVm {EM}
setfield /cc/Ca_conc Ca_base 0.0
reset

setfield /cc Vm {EM}
setfield /cc/Ca_conc Ca_base 0.0
step {PULSETIME} -t

setfield /cc Vm {EM}
setfield /cc/Ca_conc Ca_base 2e-3
step {PULSETIME} -t

setfield /cc Vm -0.040
setfield /cc/Ca_conc Ca_base 2e-3
step {PULSETIME} -t

tab2file Vm.{sim}.plot /plot1 table -overwrite
tab2file Gk.{sim}.plot /plot2 table -overwrite

quit
