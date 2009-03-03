// In this script we send in pre-defined inputs to sensed and command
// terminals of the PID.
str simulator
if ({version} < 3.0)
    simulator="genesis"
else
    simulator="moose"
end

int NSTEPS = 100
float SIMDT = 1.0

create table command
setfield command step_mode 2 stepsize 1
call command TABCREATE {NSTEPS} 0.0 {NSTEPS}

create table sensed
setfield sensed step_mode 2 stepsize 1
call sensed TABCREATE {NSTEPS} 0.0 {NSTEPS}
int i
for( i = 0; i < NSTEPS; i = i + 1 )
    setfield command table->table[{i}] {i*2e-3}
    setfield sensed table->table[{i}] {i*1e-3}
end    

create PID pid
setfield pid gain 0.1 tau_i {SIMDT} tau_d {SIMDT/4.0} saturation 999.0

addmsg command pid CMD output
addmsg sensed pid SNS output

create table pid_cmd_rec
setfield ^ step_mode 3
call pid_cmd_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_cmd_rec INPUT cmd

create table pid_sns_rec
setfield ^ step_mode 3
call pid_sns_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_sns_rec INPUT sns

create table pid_out_rec
setfield ^ step_mode 3
call pid_out_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_out_rec INPUT output

create table pid_e_rec
setfield ^ step_mode 3
call pid_e_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_e_rec INPUT e

create table pid_deriv_rec
setfield ^ step_mode 3
call pid_deriv_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_deriv_rec INPUT e_deriv

create table pid_int_rec
setfield ^ step_mode 3
call pid_int_rec TABCREATE {NSTEPS} 0.0 1.0
addmsg pid pid_int_rec INPUT e_integral


setclock 0 {SIMDT}
setclock 1 {SIMDT}
reset
step {NSTEPS}

tab2file pid_command_{simulator}.plot command table -overwrite
tab2file pid_sensed_{simulator}.plot sensed table -overwrite
tab2file pid_out_rec_{simulator}.plot pid_out_rec table -overwrite
tab2file pid_e_rec_{simulator}.plot pid_e_rec table -overwrite
tab2file pid_int_rec_{simulator}.plot pid_int_rec table -overwrite
tab2file pid_deriv_rec_{simulator}.plot pid_deriv_rec table -overwrite
tab2file pid_cmd_rec_{simulator}.plot pid_cmd_rec table -overwrite
tab2file pid_sns_rec_{simulator}.plot pid_sns_rec table -overwrite
