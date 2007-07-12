// moose

// Evaluates Exponential / Sigmoid / Linoid curves at 'v'
// ( Curve parameters: A, B, V0 )
function calc_esl( form, v, A, B, V0 )
	int form
	float v
	float A, B, V0
	
	// EXPONENTIAL
	if ( form == 1 )
		return { A * { exp { ( v - V0 ) / B } } }
	end

	// SIGMOID
	if ( form == 2 )
		return { A / ( 1.0 + { exp { ( v - V0 ) / B  } } ) }
	end
	
	// LINOID
	if ( form == 3 )
		if ( { abs { v - V0 } } < 1e-6 )
			v = v + 1e-6
		end
		
		return { A * ( v - V0 ) / ( { exp { ( v - V0 ) / B  } } - 1.0 ) }
	end
end

function calc_Na_m_alpha( v )
	float v
	
	int form   = 3
	float A    = -1.0e5
	float B    = -0.010
	float V0   = -0.040
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function calc_Na_m_beta( v )
	float v
	
	int form   = 1
	float A    = 4.0e3
	float B    = -0.018
	float V0   = -0.065
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function calc_Na_h_alpha( v )
	float v
	
	int form   = 1
	float A    = 70.0
	float B    = -0.020
	float V0   = -0.065
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function calc_Na_h_beta( v )
	float v
	
	int form   = 2
	float A    = 1.0e3
	float B    = -0.010
	float V0   = -0.035
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function calc_K_n_alpha( v )
	float v
	
	int form   = 3
	float A    = -1.0e4
	float B    = -0.010
	float V0   = -0.055
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function calc_K_n_beta( v )
	float v
	
	int form   = 1
	float A    = 125.0
	float B    = -0.080
	float V0   = -0.065
	
	return { calc_esl { form } { v } { A } { B } { V0 } }
end

function make_compartment( path, RA, RM, CM, EM, inject, diameter, length )
	str path
	float RA, RM, CM, EM, inject, diameter, length
	
	float PI = 3.141592654
	float PI_D_L = {PI} * {diameter} * {length}
	float Ra = 4.0 * {length} * {RA} / {PI_D_L}
	float Rm = {RM} / {PI_D_L}
	float Cm = {CM} * {PI_D_L}
	
	create Compartment {path}
	setfield {path} \
		Ra {Ra} \
		Rm {Rm} \
		Cm {Cm} \
		Em {EM} \
		inject {inject} \
		diameter {diameter} \
		length {length} \
		initVm {EM}
	
	float ENa     = 0.050
	float GNa     = 1200
	float EK      = -0.077
	float GK      = 360
	float Gbar_Na = {GNa} * {PI_D_L}
	float Gbar_K  = {GK}  * {PI_D_L}
	
	create HHChannel {path}/Na
	setfield {path}/Na Ek     {ENa}
	setfield {path}/Na Gbar   {Gbar_Na}
	setfield {path}/Na Xpower 3
	setfield {path}/Na Ypower 1
	setfield {path}/Na X      0.05293250
	setfield {path}/Na Y      0.59612067
	
	create HHChannel {path}/K
	setfield {path}/K Ek      {EK}
	setfield {path}/K Gbar    {Gbar_K}
	setfield {path}/K Xpower  4
	setfield {path}/K X       0.31767695
	
	addmsg {path}/channel {path}/Na/channel
	addmsg {path}/channel {path}/K/channel
	
	float VMIN  = -0.100
	float VMAX  = 0.05
	int   NDIVS = 150
	
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
	
	float v  = VMIN
	float dv = ( VMAX - VMIN ) / NDIVS
	int i
	for ( i = 0 ; i <= NDIVS; i = i + 1 )
		setfield {path}/Na/xGate/A table[{i}] \
			{ calc_Na_m_alpha { v } }
		setfield {path}/Na/xGate/B table[{i}] \
			{ { calc_Na_m_alpha { v } } + { calc_Na_m_beta { v } } }
		setfield {path}/Na/yGate/A table[{i}] \
			{ calc_Na_h_alpha { v } }
		setfield {path}/Na/yGate/B table[{i}] \
			{ { calc_Na_h_alpha { v } } +  { calc_Na_h_beta { v } } }
		setfield {path}/K/xGate/A table[{i}] \
			{ calc_K_n_alpha { v } }
		setfield {path}/K/xGate/B table[{i}] \
			{ { calc_K_n_alpha { v } } + { calc_K_n_beta { v } } }
		
		v = v + dv
	end
end

function link_compartment(path1, path2)
	str path1, path2
	addmsg {path1}/raxial {path2}/axial
end
