// moose
// GENESIS does not have this NMDA channel implementation

// This is a demo as well as a check for NMDAChan
// Subhasis Ray, NCBS, 2010-03-15

float SIMDT = 50e-6 // second
float IODT = 50e-6 // second
float SIMLENGTH = 0.5 // second
float INJECT = 1e-10 // Ampere
float EREST_ACT = -0.065 // Volt


// Make a simple compartment
create compartment comp
setfield comp initVm {EREST_ACT} \
              Em {EREST_ACT} \
              Cm 1e-12 \
              Rm 1e9 \
              Ra 1e6

pushe comp
create NMDAChan nmda
setfield nmda Gbar 0.075e-3 \
              tau1 5e-3 \
              tau2 130e-3 \
              saturation 1e6

// addmsg ../comp/VmSrc nmda/Vm
addmsg nmda/channel ../comp/channel
pope
create randomspike spike
setfield spike min_amp 1.0 \
               max_amp 1.0 \
               rate 100 \
               reset 1 \
               resetValue 0.0

addmsg spike comp/nmda SPIKE

create table spike_table
setfield ^ step_mode 3
addmsg spike spike_table INPUT state

create table gk_table 
setfield ^ step_mode 3
addmsg comp/nmda gk_table INPUT Gk

create table vm_table
setfield ^ step_mode 3
addmsg  comp vm_table INPUT Vm

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

usclock /spike 1
useclock /#[TYPE=table] 2
echo "Going to RESET simulation"
reset
showmsg comp/nmda 
showfield comp/nmda *
echo "Finished RESET simulation"
echo "Starting simulation for " {SIMLENGTH} "seconds"
showfield comp/nmda numSynapses
step {SIMLENGTH} -time
tab2file nmda_gk.plot gk_table table  -overwrite
tab2file nmda_spike.plot spike_table table  -overwrite
tab2file nmda_vm.plot vm_table table  -overwrite

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simulation finished.
Plots written to nmda_*.plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


