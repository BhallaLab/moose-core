// moose
// genesis

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tests NMDA channel.

                                                   =Gk,Ek=>
    pre.c. =Vm=> s.g. =spike=> syn. =Gk,Ek=> m.b.           post.c.
                                                  <==Vm===
    
    pre.c.    : Presynaptic compartment
    s.g.      : Spikegen
    syn.      : Synaptic channel
    m.b.      : Mg block
    post.c.   : Postsynaptic compartment

Post-synaptic compartment is voltage clamped by default, but this can be
switched off.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

include compatibility.g

////////////////////////////////////////////////////////////////////////////////
// Global parameters
////////////////////////////////////////////////////////////////////////////////

int USE_SOLVER = 1

/*
 * Flag for voltage-clamping postsynaptic compartment. This lets us look at the
 * behaviour of the Mg block at fixed voltage values.
 */
int VOLTAGE_CLAMP_POSTSYNAPTIC_COMPARTMENT = 1

float POST_SYNAPTIC_VM_MIN = -70.0e-3
float POST_SYNAPTIC_VM_MAX = 50.0e-3
float POST_SYNAPTIC_VM_STEP = 20.0e-3

str OUTPUT_DIR = "output"

////////////////////////////////////////////////////////////////////////////////
// MODEL CONSTRUCTION
////////////////////////////////////////////////////////////////////////////////

/*
 *                                                =Gk,Ek=>
 * pre.c. =Vm=> s.g. =spike=> syn. =Gk,Ek=> m.b.           post.c.
 *                                               <==Vm===
 * 
 * pre.c.    : Presynaptic compartment
 * s.g.      : Spikegen
 * syn.      : Synaptic channel
 * m.b.      : Mg block
 * post.c.   : Postsynaptic compartment
 */

float SIMDT = 50e-6
float IODT = 50e-6
float SIMLENGTH = 0.5
float STIM_TIME = 0.05

//=====================================
// Creating objects
//=====================================

//------------------
// Compartments
//------------------

/*
 * Presynaptic compartment:
 * 
 * This compartment only sends Vm to the spikegen, so we don't need to specify
 * its biophysical properties.
 * 
 * Initializing Em and initVm, which will be used to trigger spikes (spike
 * generator threshold is set to 0.0 below).
 */
create compartment /c1
setfield /c1 Em -1.0
setfield /c1 initVm -1.0

/*
 * Postsynaptic compartment:
 * 
 * This compartment's purpose is to send Vm to the Mg-block, and hence no need
 * to specify its biophysical properties.
 * 
 * However, Rm and Cm are set:
 *     - To physiologically sensible values if one wants to let the NMDA current
 *       change the post-synaptic compartment's Vm.
 * or:
 *     - To very small values if one does not want the NMDA channel to affect
 *       the compartment. This is a hack, which will keep the compartment's
 *       Vm effectively clamped, because tau = Rm * Cm will be very very small.
 *       A cleaner way to clamp is to simply cut off the NMDA's current (Gk, Ek
 *       message) to the compartment. However this cannot be done in MOOSE,
 *       without also cutting off the compartment's Vm message to the MgBlock.
 *       Also an explicit voltage clamp circuit could be used, but that is
 *       overkill for this script.
 */
create compartment /c2

if ( VOLTAGE_CLAMP_POSTSYNAPTIC_COMPARTMENT )
	setfield /c2 Rm 1e-6
	setfield /c2 Cm 1e-6
else
	setfield /c2 Rm 1e8
	setfield /c2 Cm 1e-10
end

//------------------
// Spike generator
//------------------
create spikegen /c1/spike
setfield /c1/spike \
	thresh    0.0

//------------------
// Synaptic channel
//------------------
float Ek = 0.0
float gmax = 1e-7
float tau1 = 20e-3
float tau2 = 40e-3

create synchan /c2/syn
setfield /c2/syn \
	Ek      {Ek} \
	tau1    {tau1} \
	tau2    {tau2} \
	gmax    {gmax}

//------------------
// Magnesium block
//------------------

/*
 * From GENESIS docs for Mg_block:
 * This calculates a blocked value of Gk that is reduced from the incoming Gk by
 * a factor of:
 *     A / ( A + [Mg] * exp( -Vm / B ) )
 */

float CMg = 2                       // [Mg] in mM
float eta = 0.33                    // per mM
float gamma = 60                   // per Volt

create Mg_block /c2/syn/mgblock
setfield /c2/syn/mgblock \
	CMg      {CMg} \
	KMg_A    {1.0 / eta} \
	KMg_B    {1.0 / gamma}

//=====================================
// Connections
//=====================================

/*
 *                                                =Gk,Ek=>
 * pre.c. =Vm=> s.g. =spike=> syn. =Gk,Ek=> m.b.           post.c.
 *                                               <==Vm===
 * 
 * pre.c.    : Presynaptic compartment
 * s.g.      : Spikegen
 * syn.      : Synaptic channel
 * m.b.      : Mg block
 * post.c.   : Postsynaptic compartment
 */

addmsg    /c1                /c1/spike          INPUT      Vm
addmsg    /c1/spike          /c2/syn            SPIKE
addmsg    /c2/syn            /c2/syn/mgblock    CHANNEL    Gk Ek
addmsg    /c2/syn/mgblock    /c2                CHANNEL    Gk Ek
addmsg    /c2                /c2/syn/mgblock    VOLTAGE    Vm

////////////////////////////////////////////////////////////////////////////////
// PLOTTING
////////////////////////////////////////////////////////////////////////////////
create neutral /data

create table /data/Gk_syn
call /data/Gk_syn TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Gk_syn step_mode 3

create table /data/Gk_mgblock
call /data/Gk_mgblock TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Gk_mgblock step_mode 3

create table /data/Vm
call /data/Vm TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
setfield /data/Vm step_mode 3

//=====================================
// Record data
//=====================================
addmsg /c2/syn /data/Gk_syn INPUT Gk
addmsg /c2/syn/mgblock /data/Gk_mgblock INPUT Gk
addmsg /c2 /data/Vm INPUT Vm

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONTROL
////////////////////////////////////////////////////////////////////////////////

//=====================================
//  Clocks
//=====================================
setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {SIMDT}
setclock 3 {IODT}

useclock /data/#[TYPE=table] 3

function run_and_write( post_synaptic_vm, syn_file, mgblock_file, vm_file )
	float post_synaptic_vm
	str syn_file
	str mgblock_file
	str vm_file
	
	//=====================================
	//  Simulation
	//=====================================
	setfield /c2 Em {post_synaptic_vm}
	setfield /c2 initVm {post_synaptic_vm}
	
	reset
	step {STIM_TIME} -time
	setfield /c1 Em 1.0
	setfield /c1 Vm 1.0
	step
	setfield /c1 Em -1.0
	setfield /c1 Vm -1.0
	step {SIMLENGTH - STIM_TIME} -time
	
	////////////////////////////////////////////////////////////////////////////////
	//  Write Plots
	////////////////////////////////////////////////////////////////////////////////
	openfile {syn_file} w
	writefile {syn_file} "/newplot"
	writefile {syn_file} "/plotname Gk"
	closefile {syn_file}
	tab2file {syn_file} /data/Gk_syn table
	
	openfile {mgblock_file} w
	writefile {mgblock_file} "/newplot"
	writefile {mgblock_file} "/plotname Gk"
	closefile {mgblock_file}
	tab2file {mgblock_file} /data/Gk_mgblock table
	
	openfile {vm_file} w
	writefile {vm_file} "/newplot"
	writefile {vm_file} "/plotname Gk"
	closefile {vm_file}
	tab2file {vm_file} /data/Vm table
end

int vi = 1
str extension
if ( MOOSE )
	extension = ".moose.plot"
else
	extension = ".genesis.plot"
end
float vm = POST_SYNAPTIC_VM_MIN
while ( vm <= POST_SYNAPTIC_VM_MAX )
	str syn_file = { OUTPUT_DIR } @ "/syn.Gk." @ { vi } @ { extension }
	str mgblock_file = { OUTPUT_DIR } @ "/mgblock.Gk." @ { vi } @ { extension }
	str vm_file = { OUTPUT_DIR } @ "/c2.Vm." @ { vi } @ { extension }
	
	run_and_write { vm } { syn_file } { mgblock_file } { vm_file }
	vm = vm + POST_SYNAPTIC_VM_STEP
	vi = vi + 1
end

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.moose.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

quit
