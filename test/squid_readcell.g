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
float RUNTIME = 0.5
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

setfield /sli_shell isInteractive 0

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

/*
create Compartment /squid
setfield /squid Rm {RM}
setfield /squid Ra {RA}
setfield /squid Cm {CM}
setfield /squid Em {VLEAK}
*/
create neutral /library

create HHChannel /library/Na
setfield /library/Na Ek {VNa}
setfield /library/Na Gbar {GNa}
setfield /library/Na Xpower 3
setfield /library/Na Ypower 1
create HHGate /library/Na/m
setfield /library/Na/m power 3
create HHGate /library/Na/h
setfield /library/Na/h power 1
setfield /library/Na Xpower 3
setfield /library/Na Ypower 1

create HHChannel /library/K
setfield /library/K Ek {VK}
setfield /library/K Gbar {GK}
setfield /library/K Xpower 4
create HHGate /library/K/n
setfield /library/K/n power 4
setfield /library/K Xpower 4

addmsg /library/channel /library/Na/channel
addmsg /library/channel /library/K/channel

addmsg /library/Na/xGate /library/Na/m/gate
addmsg /library/Na/yGate /library/Na/h/gate
addmsg /library/K/xGate /library/K/n/gate


setfield /library/Na/m A.xmin {VMIN}
setfield /library/Na/m B.xmin {VMIN}
setfield /library/Na/h A.xmin {VMIN}
setfield /library/Na/h B.xmin {VMIN}
setfield /library/K/n A.xmin {VMIN}
setfield /library/K/n B.xmin {VMIN}
setfield /library/Na/m A.xmax {VMAX}
setfield /library/Na/m B.xmax {VMAX}
setfield /library/Na/h A.xmax {VMAX}
setfield /library/Na/h B.xmax {VMAX}
setfield /library/K/n A.xmax {VMAX}
setfield /library/K/n B.xmax {VMAX}
setfield /library/Na/m A.xdivs {NDIVS}
setfield /library/Na/m B.xdivs {NDIVS}
setfield /library/Na/h A.xdivs {NDIVS}
setfield /library/Na/h B.xdivs {NDIVS}
setfield /library/K/n A.xdivs {NDIVS}
setfield /library/K/n B.xdivs {NDIVS}

v = VMIN
for ( i = 0 ; i <= NDIVS; i = i + 1 )
	setfield /library/Na/m A.table[{i}] { calc_Na_m_A { v } }
	setfield /library/Na/m B.table[{i}] { { calc_Na_m_A { v } } + { calc_Na_m_B { v } } }
	setfield /library/Na/h A.table[{i}] { calc_Na_h_A { v } }
	setfield /library/Na/h B.table[{i}] { { calc_Na_h_A { v } } +  { calc_Na_h_B { v } } }
	setfield /library/K/n A.table[{i}] { calc_K_n_A { v } }
	setfield /library/K/n B.table[{i}] { { calc_K_n_A { v } } + { calc_K_n_B { v } } }
//setfield /sli_shell isInteractive 1
	echo {v} { calc_K_n_B { v } }
setfield /sli_shell isInteractive 0
	v = v + dv
//	echo {v}
end

readcell squid.p /

create Plot /Vm
addmsg /Vm/trigPlot /squid/Vm

create Plot /gNa
addmsg /gNa/trigPlot /squid/Na/Gk
create Plot /gK
addmsg /gK/trigPlot /squid/K/Gk

setclock 0 {SIMDT} 0
setclock 1 {PLOTDT} 0

useclock /squid 0
useclock /##[TYPE=Plot] 1

// Crazy hack, but the library demo does it and we need to match.
setfield /squid initVm {EREST}
reset
setfield /squid Inject 0
step 0.005 -t
setfield /squid Inject {INJECT}
step 0.040 -t
setfield /squid Inject 0
step 0.005 -t
call /##[TYPE=Plot] printIn squid_readcell.plot
setfield /sli_shell isInteractive 1
// quit
