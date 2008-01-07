//moose
// This is an almost perfect match to the old GENESIS squid model
// output with default parameter values. It simulates, as the Squid
// demo does, a stimulus of 0.1 uA starting at time 5 msec and 
// lasting for 40 msec. There is a final 5 msec after the stimulus.
// Most of this file is setting up the parameters and the HH tables.
// Later I'll implement and extended version of the HHGate that knows
// the funny X_alpha and other terms that the GENESIS version uses.



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
float GLEAK = 0.3e-3
float GK = 36e-3
float GNa = 120e-3

int N_COMPARTMENT = 10
float  CABLE_LENGTH    = 1e-3
float  DIAMETER        = 1e-6
float  LENGTH          = {CABLE_LENGTH} / {N_COMPARTMENT}
float INJECT = 0.1e-6
int iNeuronIndex

/*float RM = 424.4e3	Original squid values
float RA = 7639.44e3
float CM = 0.007854e-6*/
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01

GK = 0.282743e-3
GNa = 0.94248e-3

function calc_Na_m_A( v )
	float v

	if ( { abs { {EREST} + 0.025 - v } } < 1e-6 )
		v = v + 1e-6
	end
	
	return { ( 0.1e6 * ( {EREST} + 0.025 - v ) ) / ( { exp { ( {EREST} + 0.025 - v )/ 0.01 } } - 1.0 ) }
end

function calc_Na_m_B( v )
	float v
	return { 4.0e3 * { exp { ( {EREST} - v ) / 0.018 } } }
end

function calc_Na_h_A( v )
	float v
	return { 70.0 * { exp { ( {EREST} - v ) / 0.020 } } }
end

function calc_Na_h_B( v )
	float v
	return { 1.0e3 / ( { exp { ( 0.030 + {EREST} - v )/ 0.01 } } + 1.0 ) }
end

function calc_K_n_A( v )
	float v
	if ( { abs { 0.01 + {EREST} - v } } < 1e-6 )
		v = v + 1e-6
	end
	return { ( 1.0e4 * ( 0.01 + {EREST} - v ) ) / ( { exp { ( 0.01 + {EREST} - v )/ 0.01 } } - 1.0 ) }
end

function calc_K_n_B( v )
	float v
	return { 0.125e3 * { exp { ({EREST} - v ) / 0.08 } } }
end

function make_channel (path)
	str path
	int iIndex

	create HHChannel {path}/Na
	setfield {path}/Na Ek {VNa}
	setfield {path}/Na Gbar {GNa}
	setfield {path}/Na Xpower 3
	setfield {path}/Na Ypower 1

	create HHChannel {path}/K
	setfield {path}/K Ek {VK}
	setfield {path}/K Gbar {GK}
	setfield {path}/K Xpower 4

	addmsg {path}/channel {path}/Na/channel
	addmsg {path}/channel {path}/K/channel
	
	setfield {path}/Na/xGate/A xmin {VMIN}
	setfield {path}/Na/xGate/B xmin {VMIN}
	setfield {path}/Na/yGate/A xmin {VMIN}
	setfield {path}/Na/yGate/B xmin {VMIN}
	setfield {path}/K/xGate/A xmin {VMIN}
	setfield {path}/K/xGate/B xmin {VMIN}
	setfield {path}/Na/xGate/A xmax {VMAX}
	setfield {path}/Na/xGate/B xmax {VMAX}
	setfield {path}/Na/yGate/A xmax {VMAX}
	setfield {path}/Na/yGate/B xmax {VMAX}
	setfield {path}/K/xGate/A xmax {VMAX}
	setfield {path}/K/xGate/B xmax {VMAX}
	setfield {path}/Na/xGate/A xdivs {NDIVS}
	setfield {path}/Na/xGate/B xdivs {NDIVS}
	setfield {path}/Na/yGate/A xdivs {NDIVS}
	setfield {path}/Na/yGate/B xdivs {NDIVS}
	setfield {path}/K/xGate/A xdivs {NDIVS}
	setfield {path}/K/xGate/B xdivs {NDIVS}

	v = {VMIN}
	for ( iIndex = 0 ; iIndex <= NDIVS; iIndex = iIndex + 1 )
		setfield {path}/Na/xGate/A table[{iIndex}] { calc_Na_m_A { v } }
		setfield {path}/Na/xGate/B table[{iIndex}] { { calc_Na_m_A { v } } + { calc_Na_m_B { v } } }
		setfield {path}/Na/yGate/A table[{iIndex}] { calc_Na_h_A { v } }
		setfield {path}/Na/yGate/B table[{iIndex}] { { calc_Na_h_A { v } } +  { calc_Na_h_B { v } } }
		setfield {path}/K/xGate/A table[{iIndex}] { calc_K_n_A { v } }
		setfield {path}/K/xGate/B table[{iIndex}] { { calc_K_n_A { v } } + { calc_K_n_B { v } } }

	//	echo {v} { calc_K_n_B { v } }
		v = {v} + {dv}
	//	echo {v}
	end
end	// make_channel

function link_compartment(path1, path2)
	str path1, path2
	addmsg {path1}/raxial {path2}/axial
end

function make_compartment (path, index, inject)
	str path
	int index
	float inject
	

	float PI = 3.141592654
	float PI_D_L = {PI} * {DIAMETER} * {LENGTH}
	float Ra = 4.0 * {LENGTH} * {RA} / {PI_D_L}
	float Rm = {RM} / {PI_D_L}
	float Cm = {CM} * {PI_D_L}

	create Compartment {path}
	setfield {path} Rm {Rm}
	setfield {path} Ra {Ra}
	setfield {path} Cm {Cm}
	setfield {path} Em {VLEAK}
	setfield {path} diameter {DIAMETER}
	setfield {path} length {LENGTH}
	setfield {path} initVm {EREST}
	setfield {path} inject {inject}

	make_channel {path}

	echo "Created Compartment: "{path}

end	//make_compartment


function make_neuron (path)
	str path
	int iComptIndex

	make_compartment {path}/c1 1 0

	for(iComptIndex=2; iComptIndex <= {N_COMPARTMENT}; iComptIndex = iComptIndex + 1)
		make_compartment {path}/c{iComptIndex} {iComptIndex} 0
		link_compartment {path}/c{iComptIndex-1} {path}/c{iComptIndex}
	end

	create ParSpikeGen /channel/spkgn
	create ParSynChan /channel/syncn

	create ParTable {path}/V1
	create ParTable {path}/Vm

	setfield {path}/V1 stepmode 3
	setfield {path}/Vm stepmode 3

	setfield {path}/V1 index 1
	setfield {path}/Vm index 2

	setfield /channel/syncn "tau1" 1.0e-3
	setfield /channel/syncn "tau2" 1.0e-3
	setfield /channel/syncn "Gbar" 0.9e-3
	setfield /channel/syncn "Ek"   -0.010

	setfield /channel/spkgn "threshold" 0
	//setfield /channel/spkgn "refractT" 40e-5
	setfield /channel/spkgn "refractT" 40e-3
	setfield /channel/spkgn "amplitude" 1.0
	setfield /channel/spkgn "Vm" -0.065

	addmsg {path}/V1/inputRequest {path}/c1/Vm
	addmsg {path}/Vm/inputRequest {path}/c{N_COMPARTMENT}/Vm

	addmsg {path}/c{N_COMPARTMENT}/VmSrc /channel/spkgn/Vm
	addmsg /channel/syncn/IkSrc {path}/c1/injectMsg

	//useclock {path}/squid,{path}/squid/# 0
	useclock {path}/##[TYPE=Compartment] 0
	useclock {path}/##[TYPE=HHChannel] 0
	useclock {path}/##[TYPE=ParTable] 1
        useclock /channel/spkgn 0
        useclock /channel/syncn 0
end


setclock 0 {SIMDT} 0
setclock 1 {PLOTDT} 0
create Neutral /channel

create Neutral /neuron
make_neuron /neuron

planarconnect /channel/spkgn /channel/syncn

echo "Multi neuron model setup"

reset


planardelay /channel/syncn -fixed 0.005
planarweight /channel/syncn -fixed 1

setrank 1
setfield /neuron/c1 inject {INJECT}
setrank 0

step 1.16 -t


//step 0.160 -t
//step 1.16 -t
//setfield /neuron1/c1 inject 0
//step 0.005 -t

/*echo "Dumping plot files"
for(iNeuronIndex =1 ; iNeuronIndex < 4; iNeuronIndex = iNeuronIndex +1)
	setrank {iNeuronIndex}
	setfield /neuron/V1 print "squid.plot0"{iNeuronIndex}
	setfield /neuron/Vm print "squid.plotx"{iNeuronIndex}
end
setrank 0
echo "Plot files dumped to squid.plot*"*/

quit

