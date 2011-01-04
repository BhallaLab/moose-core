/**************************************************
 * This code creates a single compartment /soma
 * with an NMDA channel and Mg_block.
 * The purpose is to test if MOOSE and GENESIS 
 * behave similarly.
 *
 * Author: Subhasis Ray, 2011-01-03
 **************************************************/

float SIMDT = 1e-4
float PLOTDT = 1e-4
float SIMTIME = 0.5
float STABTIME = 0.2
float EM = -70e-3
float RM = 5.0
float RA = 1.0
float CM = 9e-3
float SOMA_D = 20e-6
float SOMA_L = 10e-6
float SOMA_A = 3.1416 * {SOMA_D} * {SOMA_L} 
float SOMA_X = 3.1416 * {SOMA_D} * {SOMA_D} / 4.0
int NSTEPS = {(STABTIME + SIMTIME) / SIMDT - 1}
/**************************************************
 * The function make_NMDA was written by Upi.
 **************************************************/
function make_NMDA
    float CMg = 1.2 // [Mg] in mM
    float eta = 0.28                    // per mM
    float gamma = 62                    // per Volt
    float tau1_NMDA = 20e-3             // 20 msec
    float tau2_NMDA = 40e-3             // 20 msec

    if ({exists NMDA})
            return
    end
    create synchan NMDA
    setfield NMDA \
            Ek      0.0 \
            tau1    {tau1_NMDA} \
            tau2    {tau2_NMDA} \
            gmax    { 500 * SOMA_A }
    create Mg_block NMDA/block
    setfield NMDA/block \
            CMg                     {CMg} \
            Zk                      2 \
            KMg_A                   {1.0/eta} \
            KMg_B                   {1.0/gamma}


    addmsg NMDA NMDA/block CHANNEL Gk Ek

    addfield NMDA addmsg1
    setfield NMDA addmsg1        ".. ./block VOLTAGE Vm"
    addfield NMDA addmsg2
    setfield NMDA addmsg2        "./block .. CHANNEL Gk Ek"

end

/**************************************************
 * Till here was written by Upi.
 **************************************************/


create compartment /soma
setfield /soma \
	 initVm {EM} \
	 Em {EM} \
	 Rm {RM/SOMA_A} \
	 Ra {RA * SOMA_L / SOMA_X} \
	 Cm {CM * SOMA_A}
pushe /soma
make_NMDA
pope

create spikegen /spike

setfield /spike thresh 0.5e-3 abs_refract 5e-3 output_amp 1.0

if ({version} < 3.0)
   addmsg /spike /soma/NMDA SPIKE
else
   addmsg /spike/event /soma/NMDA/synapse
   setfield /spike edgeTriggered 0
end

create pulsegen /pulse
setfield /pulse \
	 level1 1.0 \
	 delay1 10e-3 \
	 width1 10e-3

addmsg /pulse /spike INPUT output

showmsg /spike 
showfield /spike *
	 
create table /vm_table
call /vm_table TABCREATE {NSTEPS} 0 1 
setfield /vm_table step_mode 3
addmsg /soma /vm_table INPUT Vm

create table /spike_table
call /spike_table TABCREATE {NSTEPS} 0 1 
setfield /spike_table step_mode 3
addmsg /spike /spike_table INPUT state

create table /pulse_table
call /pulse_table TABCREATE {NSTEPS} 0 1 
setfield /pulse_table step_mode 3
addmsg /pulse /pulse_table INPUT output

create table /gnmda_table
call /gnmda_table TABCREATE {NSTEPS} 0 1 
setfield /gnmda_table step_mode 3
addmsg /soma/NMDA /gnmda_table INPUT Gk

create table /gmg_table
call /gmg_table TABCREATE {NSTEPS} 0 1 
setfield /gmg_table step_mode 3
addmsg /soma/NMDA/block /gmg_table INPUT Gk

showmsg /soma/NMDA

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {SIMDT}
setclock 4 {PLOTDT}

useclock /# 0
useclock /soma/# 0

reset
showfield /soma/NMDA *
showfield /soma/NMDA synapse[0].weight
setfield /soma inject 0.0
step {STABTIME} -t
setfield /soma inject 1e-11
step {SIMTIME} -t
str extension = "_moose.dat"
if ({version} < 3.0)
   extension = "_genesis.dat"
end

tab2file nmda_vm{extension} /vm_table table -overwrite -nentries {NSTEPS} 
tab2file nmda_spike{extension} /spike_table table -overwrite -nentries {NSTEPS}
tab2file nmda_pulse{extension} /pulse_table table -overwrite -nentries {NSTEPS}
tab2file nmda_gk{extension} /gnmda_table table -overwrite -nentries {NSTEPS}
tab2file nmda_mg{extension} /gmg_table table -overwrite -nentries {NSTEPS}
echo "Finished simulation for {SIMTIME+STABTIME}. Saved data in:"
echo nmda_vm{extension} - compartment Vm
echo nmda_spike{extension} - spike input to NMDA channel
echo nmda_gk{extension} - conductance of the NMDA channel
echo nmda_mg{extension} - current after scaling by the Mg_block.
