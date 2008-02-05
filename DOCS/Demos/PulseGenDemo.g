float RUNTIME = 100.0
float SIMDT = 1
float PLOTDT = 1
create PulseGen /pulser
setfield /pulser level1 50.0 width1 3.0 delay1 5.0 level2 -20.0 width2 5.0 delay2 8.0 baselevel 10.0 trig_mode 0
create Table /tab
call tab TABCREATE { RUNTIME / PLOTDT } 0 1
setfield /tab step_mode 3
addmsg /pulser /tab INPUT output
//addmsg /pulser/outputSrc /tab/inputRequest 
setclock 0 {SIMDT}
//setclock 1 {PLOTDT}
useclock /pulser,/tab 0
//useclock /tab 1
step {RUNTIME} -t
tab2file pulse.plot tab table -nentries {RUNTIME/PLOTDT - 1} -overwrite
quit
