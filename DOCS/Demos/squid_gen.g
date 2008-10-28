//moose
// This is an almost perfect match to the old GENESIS squid model
// output with default parameter values. It simulates, as the Squid
// demo does, a stimulus of 0.1 uA starting at time 5 msec and 
// lasting for 40 msec. There is a final 5 msec after the stimulus.
// Most of this file is setting up the parameters and the HH tables.
// Later I'll implement and extended version of the HHGate that knows
// the funny X_alpha and other terms that the GENESIS version uses.

int i

float VMIN = -0.1
float VMAX = 0.05
int NDIVS = 150
float v = VMIN
float dv = ( VMAX - VMIN ) / NDIVS
float SIMDT = 1e-5
float PLOTDT = 1e-4
float RUNTIME = 0.05
float EREST = -0.07
float VLEAK = EREST + 0.010613
float VK = EREST -0.012
float VNa = EREST + 0.115
float RM = 424.4e3
float RA = 7639.44e3
float GLEAK = 0.3e-3
float GK = 36e-3
float GNa = 120e-3
float CM = 0.007854e-6
float INJECT = 0.1e-6

GK = 0.282743e-3
GNa = 0.94248e-3

function calc_Na_m_A( v )
	float v

	if ( { abs { EREST + 0.025 - v } } < 1e-6 )
		v = v + 1e-6
	end
	
	return { ( 0.1e6 * ( EREST + 0.025 - v ) ) / ( { exp { ( EREST + 0.025 - v )/ 0.01 } } - 1.0 ) }
end

function calc_Na_m_B( v )
	float v
	return { 4.0e3 * { exp { ( EREST - v ) / 0.018 } } }
end

function calc_Na_h_A( v )
	float v
	return { 70.0 * { exp { ( EREST - v ) / 0.020 } } }
end

function calc_Na_h_B( v )
	float v
	return { 1.0e3 / ( { exp { ( 0.030 + EREST - v )/ 0.01 } } + 1.0 ) }
end

function calc_K_n_A( v )
	float v
	if ( { abs { 0.01 + EREST - v } } < 1e-6 )
		v = v + 1e-6
	end
	return { ( 1.0e4 * ( 0.01 + EREST - v ) ) / ( { exp { ( 0.01 + EREST - v )/ 0.01 } } - 1.0 ) }
end

function calc_K_n_B( v )
	float v
	return { 0.125e3 * { exp { (EREST - v ) / 0.08 } } }
end

create compartment /squid
setfield /squid Rm {RM}
setfield /squid Ra {RA}
setfield /squid Cm {CM}
setfield /squid Em {VLEAK}

create tabchannel /squid/Na
setfield /squid/Na Ek {VNa}
setfield /squid/Na Gbar {GNa}
setfield /squid/Na Xpower 3
setfield /squid/Na Ypower 1

create tabchannel /squid/K
setfield /squid/K Ek {VK}
setfield /squid/K Gbar {GK}
setfield /squid/K Xpower 4

// addmsg /squid/channel /squid/Na/channel
// addmsg /squid/channel /squid/K/channel

addmsg /squid /squid/Na VOLTAGE Vm
addmsg /squid/Na /squid CHANNEL Gk Ek

addmsg /squid /squid/K VOLTAGE Vm
addmsg /squid/K /squid CHANNEL Gk Ek

create table /Vm
call /Vm TABCREATE {RUNTIME / PLOTDT} 0 1
setfield /Vm step_mode 3
addmsg /squid /Vm INPUT Vm
// addmsg /Vm/inputRequest /squid/Vm




// create Plot /Vm
// addmsg /Vm/trigPlot /squid/Vm

// create Plot /gNa
// addmsg /gNa/trigPlot /squid/Na/Gk
// create Plot /gK
// addmsg /gK/trigPlot /squid/K/Gk

call /squid/Na TABCREATE X {NDIVS} {VMIN} {VMAX}
call /squid/Na TABCREATE Y {NDIVS} {VMIN} {VMAX}
call /squid/K TABCREATE X {NDIVS} {VMIN} {VMAX}

/*
setfield /squid/Na/xGate/A xmin {VMIN}
setfield /squid/Na/xGate/B xmin {VMIN}
setfield /squid/Na/yGate/A xmin {VMIN}
setfield /squid/Na/yGate/B xmin {VMIN}
setfield /squid/K/xGate/A xmin {VMIN}
setfield /squid/K/xGate/B xmin {VMIN}
setfield /squid/Na/xGate/A xmax {VMAX}
setfield /squid/Na/xGate/B xmax {VMAX}
setfield /squid/Na/yGate/A xmax {VMAX}
setfield /squid/Na/yGate/B xmax {VMAX}
setfield /squid/K/xGate/A xmax {VMAX}
setfield /squid/K/xGate/B xmax {VMAX}
setfield /squid/Na/xGate/A xdivs {NDIVS}
setfield /squid/Na/xGate/B xdivs {NDIVS}
setfield /squid/Na/yGate/A xdivs {NDIVS}
setfield /squid/Na/yGate/B xdivs {NDIVS}
setfield /squid/K/xGate/A xdivs {NDIVS}
setfield /squid/K/xGate/B xdivs {NDIVS}
*/

v = VMIN
for ( i = 0 ; i <= NDIVS; i = i + 1 )
	setfield /squid/Na X_A->table[{i}] { calc_Na_m_A { v } }
	setfield /squid/Na X_B->table[{i}] { { calc_Na_m_A { v } } + { calc_Na_m_B { v } } }
	setfield /squid/Na Y_A->table[{i}] { calc_Na_h_A { v } }
	setfield /squid/Na Y_B->table[{i}] { { calc_Na_h_A { v } } +  { calc_Na_h_B { v } } }
	setfield /squid/K X_A->table[{i}] { calc_K_n_A { v } }
	setfield /squid/K X_B->table[{i}] { { calc_K_n_A { v } } + { calc_K_n_B { v } } }
//	echo {v} { calc_K_n_B { v } }
	v = v + dv
//	echo {v}
end

// Use a pulsegen object to apply current injection
create pulsegen /pulse
setfield /pulse level1 {INJECT} delay1 0.005 width1 0.04
addmsg /pulse /squid INJECT output
setclock 0 {SIMDT}
setclock 1 {PLOTDT}

useclock /squid,/squid/#,/pulse 0
// useclock /##[TYPE=table] 1
useclock /Vm 1


setfield /squid initVm {EREST}
reset
// Setting the injection current explicitly is replaced by pulsegen object
// setfield /squid inject 0
// step 0.005 -t
// setfield /squid inject {INJECT}
// step 0.040 -t
// setfield /squid inject 0
// step 0.005 -t
step 0.05 -t
int nsteps = RUNTIME / PLOTDT - 1
tab2file squid_gen.plot /Vm table -nentries {nsteps} -overwrite
// setfield /Vm print "squid.plot"
// call /##[TYPE=Plot] printIn squid.plot
// quit
