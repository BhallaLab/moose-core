// genesis
include graphics_funcs.g 

include squid_funcs.g 
include squid_graphs.g 
include squid_electronics.g 
include squid_forms.g 
include squid_sched.g 
include phsplot.g // added for Squid_tut

include tut_forms.g // added for Squid_tut
include setcolors.g // functions to change the default colors

// start up Xodus

xcolorscale hot

//-------
// SETUP 
//-------
squid_cmpt /axon 500 500
copy /axon /target
create spikegen /axon/spike
		setfield /axon/spike thresh 60 abs_refract 20 output_amp 1

create synchan /target/synapse
setfield /target/synapse tau1 1 tau2 1 \
	gmax {getfield /axon/Na gbar} Ek {getfield /axon/Na Ek}

add_squid_graphs /axon
add_squid_electronics
connect_squid_electronics /axon
add_squid_forms
// added for Squid_tut
add_squid_phaseplot
// added for Squid_tut
add_squid_popups

// Set up the synapse
delete /target/Na
delete /target/K


addmsg /axon /axon/spike INPUT Vm
addmsg /axon/spike /target/synapse SPIKE
addmsg /target /target/synapse VOLTAGE Vm
addmsg /target/synapse /target CHANNEL Gk Ek
setfield /target/synapse synapse[0].delay 10 synapse[0].weight 1.0
/*
reset
step 2
call /target/synapse RECALC
reset
step 2
call /target/synapse RECALC
reset
*/
// setconn /axon/spike:0 delay 10 weight 1


//-----------------
// GRAPH MESSAGES
//-----------------
addmsg /axon /axon/graphs/Vm PLOT Vm *Vm *red
addmsg /target /axon/graphs/Vm PLOT Vm *targetVm *red
addmsg /target/synapse /axon/graphs/Gk PLOT Gk *synapse *black
addmsg /Vclamp /axon/graphs/Vm PLOT output *command *blue

addmsg /Iclamp /axon/graphs/inj PLOT output *Iclamp *red
addmsg /PID /axon/graphs/inj PLOT output *Vclamp *blue


addmsg /axon/Na /axon/graphs/Gk PLOT Gk *Na *blue
addmsg /axon/K /axon/graphs/Gk PLOT Gk *K *red

addmsg /axon/Na /axon/graphs/Ik PLOTSCALE Ik *Na *blue -1 0
addmsg /axon/K /axon/graphs/Ik PLOTSCALE Ik *K *red -1 0



//-----------------
// INITIALIZATION
//-----------------
upicolors  // set the widget colors to something bright
xshow /forms/control
//added by Ed Vigmond
xshow /forms/exconcen
ce /forms/control
call tstep B1DOWN
call dt B1DOWN
call clamp_mode B1DOWN
call reset B1DOWN
