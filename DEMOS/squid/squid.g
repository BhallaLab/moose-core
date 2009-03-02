str SIMULATOR
if ( {version} < 3.0 )
	SIMULATOR = "genesis"
else
    SIMULATOR = "moose"
end

float VMIN = -0.1
float VMAX = 0.05
int NDIVS = 150
float dv = ( VMAX - VMIN ) / NDIVS
float SIMTIME = 0.05
float SIMDT = 1e-6
float PLOTDT = 1e-6
float EREST = -0.07
float VLEAK = EREST + 0.010613
float VK = EREST - 0.012
float VNa = EREST + 0.115
float RM = 424.4e3
float RA = 7639.44e3
//float GLEAK = 0.3e-3
float GK = 36e-3
float GNa = 120e-3
float CM = 0.007854e-6
float INJECT = 0.1e-6
int PLOTSTEPS = {SIMTIME / PLOTDT}
GK = 0.282743e-3
GNa = 0.94248e-3
int CLAMP = 0 // Voltage Clamp
////////////////////////////////////////////////////////////////////////////////
// Utility functions to calculate m and h parameters
////////////////////////////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////////////////////////////
// Functions to create the channels
////////////////////////////////////////////////////////////////////////////////
function make_Na(compartment)
    str compartment
    create tabchannel {compartment}/Na
    setfield ^ Ek {VNa} Gbar {GNa} Xpower 3 Ypower 1
    addmsg {compartment} {compartment}/Na VOLTAGE Vm
    addmsg {compartment}/Na {compartment} CHANNEL Gk Ek
    call {compartment}/Na TABCREATE X {NDIVS} {VMIN} {VMAX}
    call {compartment}/Na TABCREATE Y {NDIVS} {VMIN} {VMAX}

    float v = VMIN
    int i
    for ( i = 0 ; i <= NDIVS; i = i + 1 )
        setfield {compartment}/Na X_A->table[{i}] { calc_Na_m_A { v } }
		setfield {compartment}/Na X_B->table[{i}] { { calc_Na_m_A { v } } + { calc_Na_m_B { v } } }
		setfield {compartment}/Na Y_A->table[{i}] { calc_Na_h_A { v } }
		setfield {compartment}/Na Y_B->table[{i}] { { calc_Na_h_A { v } } +  { calc_Na_h_B { v } } }
		v = v + dv
    end
end


function make_K(compartment)
    str compartment
    create tabchannel {compartment}/K
    setfield {compartment}/K Ek {VK} Gbar {GK} Xpower 4
    addmsg {compartment}   {compartment}/K VOLTAGE Vm
    addmsg {compartment}/K {compartment} CHANNEL Gk Ek
    call {compartment}/K TABCREATE X {NDIVS} {VMIN} {VMAX}
    float v = VMIN
    int i = 0
    for ( i = 0 ; i <= NDIVS; i = i + 1 )
        setfield {compartment}/K X_A->table[{i}] { calc_K_n_A { v } }
        setfield {compartment}/K X_B->table[{i}] { { calc_K_n_A { v } } + { calc_K_n_B { v } } }
        v = v + dv
    end

end

////////////////////////////////////////////////////////////////////////////////
// Function to create the compartment
////////////////////////////////////////////////////////////////////////////////
function create_containers
    if ((!{exists /model}))
        create neutral /model
    end
    if((!{exists /data}))
        create neutral /data
    end
    if ((!{exists /elec}))
        create neutral /elec
    end
end
function create_compartment
    create_containers
    str path = "/model/squid"
    create compartment {path}
    echo "Created compartment" {path}
    setfield {path} Rm {RM} Ra {RA} Cm {CM} Em {VLEAK} initVm {EREST}
    make_Na {path}
    make_K {path}
end
////////////////////////////////////////////////////////////////////////////////
// Utility functions to add electronics
////////////////////////////////////////////////////////////////////////////////
function make_electronics
    create_containers
    str path = "/elec/pulsegen"
	if (({exists {path}}))
		return
	end
    create pulsegen {path}

    str path = "/elec/vclamp"
	create diffamp {path}
    echo "Created diffamp for vclamp" {path}
	setfield ^ saturation 999.0 \	// unitless I hope
		   gain 1.0	// 1/R  from the lowpass filter input

    str path = "/elec/iclamp"
    create diffamp {path}
    setfield ^ saturation 1e6 gain 0.0
    echo "Created  diffamp for Iclamp" {path}

    path = "/elec/lowpass"
	create RC {path}
    echo "Created RC" {path}
	setfield ^ R 1.0 C 50e-6	// ohms and farads; for a tau of 50 us

    path = "/elec/PID"
	create PID {path}
    echo "Created PID" {path}
	setfield ^ gain 1e-5 \ 	// 10/Rinput of cell
		tau_i 20e-6 \ 	// seconds
		tau_d 5e-6 saturation 999.0

    addmsg /elec/pulsegen /elec/iclamp PLUS output
    addmsg /elec/pulsegen /elec/lowpass INJECT output
	addmsg /elec/lowpass /elec/vclamp PLUS state
    addmsg /elec/vclamp /elec/PID CMD output

end
function connect_elec
    addmsg /elec/iclamp /model/squid INJECT output 
    addmsg /elec/PID /model/squid INJECT output
    addmsg /model/squid /elec/PID SNS Vm
end

function connect_iclamp
    echo "Connecting Iclamp"
    setfield /elec/iclamp gain 1.0
    setfield /elec/PID gain 0.0
    setfield /elec/pulsegen trig_mode 0 delay1 10e-3 baselevel 0.1e-11 width1 30e-3 level1 0.1e-6 delay2 1e6 level2 0

end
    
function connect_vclamp(pre_time, pre_v, clamp_time, clamp_v)
    echo "Connecting Vclamp"
    setfield /elec/iclamp gain 0.0
    setfield /elec/vclamp gain 1.0
    setfield /elec/PID gain 0.5e-3
    setfield /elec/pulsegen trig_mode 0 delay1 {pre_time} baselevel {pre_v} width1 {clamp_time} level1 {clamp_v} delay2 1e6 level2 0
end

function create_tables
    create_containers
    pushe /data
    // recording table for membrane potential Vm
    create table /data/Vm
    setfield ^ step_mode 3
    call /data/Vm TABCREATE {PLOTSTEPS} {VMIN} {VMAX}
    // recording table for injection current 'inject'
    create table /data/inject
    setfield ^ step_mode 3
    call /data/inject TABCREATE {PLOTSTEPS} -1.0 1.0
    // recording table for Na channel current INa
    create table /data/ina
    setfield ^ step_mode 3
    call /data/ina TABCREATE {PLOTSTEPS} -1.0 1.0
    // recording table for K channel current IK
    create table /data/ik
    setfield ^ step_mode 3
    call /data/ik TABCREATE {PLOTSTEPS} -1.0 1.0
    // recording table for Na channel conductance GNa
    create table /data/gna
    setfield ^ step_mode 3
    call /data/gna TABCREATE {PLOTSTEPS} -1.0 1.0
    // recording table for K channel conductance GK
    create table /data/gk
    setfield ^ step_mode 3
    call /data/gk TABCREATE {PLOTSTEPS} -1.0 1.0
    // recording table for output of the RC filter
    create table /data/lowpass
    setfield ^ step_mode 3
    call /data/lowpass TABCREATE {PLOTSTEPS} -1.0 1.0
    create table /data/vclamp
    setfield ^ step_mode 3
    call /data/vclamp TABCREATE {PLOTSTEPS} -1.0 1.0
    // save intermediate values from PID for debugging
    create table /data/error
    setfield ^ step_mode 3
    call /data/error TABCREATE {PLOTSTEPS} -1.0 1.0
    create table /data/eintegral
    setfield ^ step_mode 3
    call /data/eintegral TABCREATE {PLOTSTEPS} -1.0 1.0
    create table /data/ederiv
    setfield ^ step_mode 3
    call /data/ederiv TABCREATE {PLOTSTEPS} -1.0 1.0
    
    create table /data/pulsegen 
    setfield ^ step_mode 3
    call /data/pulsegen TABCREATE {PLOTSTEPS} 0.0 1.0
    pope
end

function connect_tables
    str comp_path = "/model/squid"
    echo "Connecting tables for" {comp_path}
    addmsg {comp_path} /data/Vm INPUT Vm
    echo "Added" {comp_path} "to /data/Vm"
    addmsg {comp_path} /data/inject INPUT Im
    echo "Added" {comp_path} "to /data/inject"
    addmsg {comp_path}/Na /data/ina INPUT Ik
    addmsg {comp_path}/K /data/ik INPUT Ik
    addmsg {comp_path}/Na /data/gna INPUT Gk
    addmsg {comp_path}/K /data/gk INPUT Gk
    addmsg /elec/pulsegen /data/pulsegen INPUT output
    addmsg /elec/lowpass /data/lowpass INPUT state
    addmsg /elec/vclamp /data/vclamp INPUT output
    addmsg /elec/PID /data/error INPUT e
    addmsg /elec/PID /data/eintegral INPUT e_integral
    addmsg /elec/PID /data/ederiv INPUT e_deriv

end

function setup_model
    create_compartment
    make_electronics
    create_tables
    connect_tables
    connect_elec
end

function schedule
    if ({version} < 3.0)
        setclock 0 {SIMDT}
        setclock 1 {PLOTDT}
        useclock /model/##,/elec/## 0
        useclock /data/## 1
    else
        setclock 0 {SIMDT} 0
        setclock 1 {SIMDT} 0
        setclock 2 {PLOTDT} 0
        useclock /model/##,/elec/# 0 
        useclock /model/squid 1 init
        useclock /data/# 2
    end
end

function save_plots
    str tab
    foreach tab ({el /data/#})
        str path = {SIMULATOR} @ "_" @ {getfield {tab} name} @ ".plot"
        tab2file {path} {tab} table -overwrite
        echo "Dumped" {tab} "in" {path}
    end
    tab2file {{SIMULATOR} @ "_alpha_m.plot"} /model/squid/Na X_A->table -overwrite
    tab2file {{SIMULATOR} @ "_tau_m.plot"} /model/squid/Na X_B->table -overwrite
    tab2file {{SIMULATOR} @ "_alpha_h.plot"} /model/squid/Na Y_A->table -overwrite   
    tab2file {{SIMULATOR} @ "_tau_h.plot"} /model/squid/Na Y_B->table -overwrite
    tab2file {{SIMULATOR} @ "_alpha_n.plot"} /model/squid/K X_A->table -overwrite
    tab2file {{SIMULATOR} @ "_tau_n.plot"} /model/squid/K X_B->table -overwrite
end

function run_sim(clamp_mode)
    int clamp_mode
    setup_model
    schedule
    if ({clamp_mode} == 0)
        connect_vclamp 10e-3 {EREST} 20e-3 50e-3
    else
        connect_iclamp
    end
    reset
    echo "Stepping for" {SIMTIME} "s"
    step {SIMTIME} -t
 
    save_plots
end

run_sim 0
//quit
