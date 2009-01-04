// This is a test and demo for series RC circuit.
float SIMTIME = 3.0
float SIMDT = 5e-2

int NSTEPS = SIMTIME / SIMDT
create RC rc_circuit
setfield ^ R 1e3 C 1e-3 V0 1e-3 inject 2e-3
create table vc_table
setfield ^ step_mode 3
call ^ TABCREATE {NSTEPS+1} 0 1
addmsg rc_circuit vc_table INPUT state
setclock 0 {SIMDT}
reset
step {SIMTIME} -t
tab2file RCDemo.plot vc_table table -overwrite
