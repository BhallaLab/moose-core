// moose
// genesis

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

function embed_Na( compartment )
	str compartment
	pushe {compartment}
	
	if ( { exists Na } )
		pope
		return
	end
	
	float PI = 3.141592654
	float diameter = { getfield dia }
	float length = { getfield len }
	float area = { PI * diameter * length }
	
	float Ek = 0.050
	float Gbar = { 1200 * area }
	
	create tabchannel Na
	setfield ^ \
		Ek     {Ek} \
		Gbar   {Gbar} \
		Xpower 3 \
		Ypower 1
	
	float VMIN  = -0.100
	float VMAX  = 0.05
	int   NDIVS = 150
	
	call Na TABCREATE X {NDIVS} {VMIN} {VMAX}
	call Na TABCREATE Y {NDIVS} {VMIN} {VMAX}
	
	float v  = VMIN
	float dv = ( VMAX - VMIN ) / NDIVS
	int i
	for ( i = 0 ; i <= NDIVS; i = i + 1 )
		// X gate, A table ( A(v) = alpha(v) )
		setfield Na X_A->table[{i}] \
			{ calc_Na_m_alpha { v } }
		
		// X gate, B table ( B(v) = alpha(v) + beta(v) )
		setfield Na X_B->table[{i}] \
			{ { calc_Na_m_alpha { v } } + { calc_Na_m_beta { v } } }
		
		// Y gate, A table ( A(v) = alpha(v) )
		setfield Na Y_A->table[{i}] \
			{ calc_Na_h_alpha { v } }
		
		// Y gate, B table ( B(v) = alpha(v) + beta(v) )
		setfield Na Y_B->table[{i}] \
			{ { calc_Na_h_alpha { v } } +  { calc_Na_h_beta { v } } }
		
		v = v + dv
	end
	
	addmsg . Na VOLTAGE Vm
	addmsg Na . CHANNEL Gk Ek
	
	pope
end

function embed_K( compartment )
	str compartment
	pushe {compartment}
	
	if ( { exists K } )
		pope
		return
	end
	
	float PI = 3.141592654
	float diameter = { getfield dia }
	float length = { getfield len }
	float area = { PI * diameter * length }
	
	float Ek = -0.077
	float Gbar = { 360 * area }
	
	create tabchannel K
	setfield ^ \
		Ek     {Ek} \
		Gbar   {Gbar} \
		Xpower 4
	
	float VMIN  = -0.100
	float VMAX  = 0.05
	int   NDIVS = 150
	
	call K TABCREATE X {NDIVS} {VMIN} {VMAX}
	
	float v  = VMIN
	float dv = ( VMAX - VMIN ) / NDIVS
	int i
	for ( i = 0 ; i <= NDIVS; i = i + 1 )
		// X gate, A table ( A(v) = alpha(v) )
		setfield K X_A->table[{i}] \
			{ calc_K_n_alpha { v } }
		
		// X gate, B table ( B(v) = alpha(v) + beta(v) )
		setfield K X_B->table[{i}] \
			{ { calc_K_n_alpha { v } } + { calc_K_n_beta { v } } }
		
		v = v + dv
	end
	
	addmsg . K VOLTAGE Vm
	addmsg K . CHANNEL Gk Ek
	
	pope
end
