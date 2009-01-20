// This is a test and demo for series RC circuit.
float SIMTIME = 3.0
float SIMDT = 5e-3

int NSTEPS = SIMTIME / SIMDT
create RC rc_circuit
setfield ^ R 1e3 C 1e-3 V0 1e-3 inject 2e-3

create pulsegen pulse_gen
setfield ^ delay1 1.0 level1 4e-3 width1 0.5
addmsg pulse_gen rc_circuit INJECT output

create table vc_table
setfield ^ step_mode 3
call ^ TABCREATE {NSTEPS+1} 0 1
addmsg rc_circuit vc_table INPUT state

setclock 0 {SIMDT}
reset
step {SIMTIME} -t
if ({version} < 3.0)
    tab2file rcdemo_genesis.plot vc_table table -overwrite
else
    tab2file rcdemo_moose.plot vc_table table -overwrite
end
