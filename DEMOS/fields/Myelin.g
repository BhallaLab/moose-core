// moose
// genesis


echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Much like the axon demo, a linear excitable cell is created using readcell.   %
% Integration is done using the Hines' method.                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

////////////////////////////////////////////////////////////////////////////////
// COMPATIBILITY (between MOOSE and GENESIS)
////////////////////////////////////////////////////////////////////////////////
include compatibility.g


////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////
float SIMDT = 50e-6
float IODT = 100e-6
float SIMLENGTH = 0.25
float INJECT = 1e-10
float EREST_ACT = -0.065

include chan.g

ce /library
	make_Na_mit_usb
	make_K_mit_usb
ce /

//=====================================
//  Create cells
//=====================================
readcell myelin2.p /axon


////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /plot

create table /plot/Vm
call /plot/Vm TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Vm step_mode 3
addmsg /axon/n20/i10 /plot/Vm INPUT Vm
useclock /plot/Vm 2

create table /plot/Gbar
call /plot/Gbar TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Gbar step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Gbar INPUT Gbar
useclock /plot/Gbar 2

create table /plot/Ek
call /plot/Ek TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Ek step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Ek INPUT Ek
useclock /plot/Ek 2

create table /plot/Gk
call /plot/Gk TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Gk step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Gk INPUT Gk
useclock /plot/Gk 2

create table /plot/Ik
call /plot/Ik TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Ik step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Ik INPUT Ik
useclock /plot/Ik 2

create table /plot/X
call /plot/X TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/X step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/X INPUT X
useclock /plot/X 2

create table /plot/Y
call /plot/Y TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Y step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Y INPUT Y
useclock /plot/Y 2

create table /plot/Z
call /plot/Z TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /plot/Z step_mode 3
addmsg /axon/n20/Na_mit_usb /plot/Z INPUT Z
useclock /plot/Z 2

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
if ( MOOSE )
	setclock 0 {SIMDT} 0
	setclock 1 {SIMDT} 1
	setclock 2 {IODT} 0
else
	setclock 0 {SIMDT}
	setclock 1 {SIMDT}
	setclock 2 {IODT}
end

//=====================================
//  Stimulus
//=====================================
setfield /axon/soma inject {INJECT}

//=====================================
//  Solvers
//=====================================
if ( GENESIS )
	create hsolve /axon/solve
	setfield /axon/solve \
		path /axon/##[TYPE=symcompartment],/axon/##[TYPE=compartment] \
		chanmode 1
	call /axon/solve SETUP
	setmethod 11
end

//=====================================
//  Simulation
//=====================================
reset
step {SIMLENGTH} -time


////////////////////////////////////////////////////////////////////////////////
//  Write Plots
////////////////////////////////////////////////////////////////////////////////
str filename
str extension
if ( MOOSE )
	extension = ".moose"
else
	extension = ".genesis"
end

filename = "Vm.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Vm"
closefile {filename}
tab2file {filename} /plot/Vm table

filename = "Gbar.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Gbar"
closefile {filename}
tab2file {filename} /plot/Gbar table

filename = "Ek.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Ek"
closefile {filename}
tab2file {filename} /plot/Ek table

filename = "Gk.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Gk"
closefile {filename}
tab2file {filename} /plot/Gk table

filename = "Ik.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Ik"
closefile {filename}
tab2file {filename} /plot/Ik table

filename = "X.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname X"
closefile {filename}
tab2file {filename} /plot/X table

filename = "Y.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Y"
closefile {filename}
tab2file {filename} /plot/Y table

filename = "Z.plot" @ extension
openfile {filename} w
writefile {filename} "/newplot"
writefile {filename} "/plotname Z"
closefile {filename}
tab2file {filename} /plot/Z table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots written to axon.*.plot.                                                   %
% If you have gnuplot, run 'gnuplot myelin.gnuplot' to view the graphs.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
