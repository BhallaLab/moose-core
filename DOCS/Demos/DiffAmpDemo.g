//This is to test the difference amplifier object
create diffamp amp
// setfield ^ gain 1.5
create pulsegen pulse_a
setfield ^ delay1 1 level1 1 width1 4
create pulsegen pulse_b
setfield ^ delay1 0.5 level1 1.5 width1 2
create pulsegen pulse_c
setfield ^ delay1 2 level1 2 width1 2 
create table amp_table
setfield ^ stepmode 3
create table pulse_a_table 
setfield ^ stepmode 3
create table pulse_b_table 
setfield ^ stepmode 3
create table pulse_c_table 
setfield ^ stepmode 3

addmsg pulse_a amp PLUS output
addmsg pulse_b amp MINUS output
addmsg pulse_c amp MINUS output
addmsg amp amp_table INPUT output
addmsg pulse_a pulse_a_table INPUT output
addmsg pulse_b pulse_b_table INPUT output
addmsg pulse_c pulse_c_table INPUT output

reset 
setclock 0 1e-2
// useclock pulse_a,pulse_b,pulse_c,amp,output_table 0
step 6 -t

tab2file diffamp_out.plot amp_table table -overwrite
tab2file diffamp_input_plus.plot pulse_a_table table -overwrite
tab2file diffamp_input_minus1.plot pulse_b_table table -overwrite
tab2file diffamp_input_minus2.plot pulse_c_table table -overwrite

quit
