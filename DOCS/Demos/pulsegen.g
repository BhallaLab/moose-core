echo "Testing PulseGen"
// Below code has been copied from PulseGenDemo.g for genesis
//genesis - demonstration of pulsegen object
float RUNTIME = 200

float SIMDT = 1
int STEPS = {RUNTIME/SIMDT - 1}
create pulsegen /pulse0
setfield ^ level1 50.0 width1 3.0 delay1 5.0 level2 -20.0 width2 5.0  \
    delay2 8.0 baselevel 10.0 trig_mode 0

create pulsegen /trig
setfield ^ level1 20.0 width1 1.0 delay1 5.0 width2 30.0

create pulsegen /pulse1
setfield ^ level1 50.0 width1 3.0 delay1 5.0 level2 -20.0 width2 5.0  \
    delay2 8.0 baselevel 10.0 trig_mode 1

addmsg /trig /pulse1 INPUT output

create pulsegen /gate
setfield ^ level1 20.0 width1 30.0 delay1 15.0 width2 30.0

create pulsegen /pulse2
setfield ^ level1 50.0 width1 3.0 delay1 5.0 level2 -20.0 width2 5.0  \
    delay2 8.0 baselevel 10.0 trig_mode 2

addmsg /gate /pulse2 INPUT output

// We skip the graphics stuff and dump the results in files instead
create table /plot0
call /plot0 TABCREATE {STEPS} 0 1
setfield /plot0 step_mode 3
addmsg /pulse0 /plot0 INPUT output

create table /plot1
call /plot1 TABCREATE {STEPS} 0 1
setfield /plot1 step_mode 3
addmsg /pulse1 /plot1 INPUT output

create table /plot2
call /plot2 TABCREATE {STEPS} 0 1
setfield /plot2 step_mode 3
addmsg /pulse2 /plot2 INPUT output

create table /plot_gate
call /plot_gate TABCREATE {STEPS} 0 1
setfield /plot_gate step_mode 3
addmsg /gate /plot_gate INPUT output

create table /plot_trig
call /plot_trig TABCREATE {STEPS} 0 1
setfield /plot_trig step_mode 3
addmsg /trig /plot_trig INPUT output

setclock 0 {SIMDT}
setclock 1 {SIMDT}
useclock /pulse0,/pulse1,/pulse2,/trig,/gate 0
useclock /plot0,/plot1,/plot2,/plot_trig,/plot_gate 1
reset
step {RUNTIME} -t
tab2file pulse0.plot /plot0 table -nentries {STEPS} -overwrite
tab2file pulse1.plot /plot1 table -nentries {STEPS} -overwrite
tab2file pulse2.plot /plot2 table -nentries {STEPS} -overwrite
tab2file gate.plot /plot_gate table -nentries {STEPS} -overwrite
tab2file trig.plot /plot_trig table -nentries {STEPS} -overwrite
echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Present curves are in pulse0.plot, pulse1.plot, pulse2.plot, trig.plot and 
% gate.plot.                    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
