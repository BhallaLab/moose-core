TITLE Sodium transient current for RD Traub et al 2003, 2005

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)
	fastNashift init to 0 and removed from arg modification Tom Morse 3/8/2006
	(for Traub et al 2005)
ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX naf
	USEION na READ ena WRITE ina
	RANGE gbar, ina,m, h, df, fastNa_shift, a, b, c, d, minf, mtau
}
PARAMETER { 
	fastNa_shift = 0: orig -3.5 (mV)
	a = 0 (1)
	b = 0 (1)
	c = 0 (1)
	d = 0 (1)
	gbar = 0.0 	   (mho/cm2)
	v (mV) ena 		   (mV)  
} 
ASSIGNED { 
	ina 		   (mA/cm2) 
	minf hinf 	   (1)
	mtau (ms) htau 	   (ms)
	df	(mV)
} 
STATE {
	m h
}
BREAKPOINT { 
	SOLVE states METHOD cnexp
	ina = gbar * m * m * m * h * ( v - ena ) 
	df = v - ena
} 
INITIAL { 
	settables( v )
	m = minf
	m = 0
	h  = hinf
} 
DERIVATIVE states { 
	settables( v ) 
	m' = ( minf - m ) / mtau 
	h' = ( hinf - h ) / htau
}

UNITSOFF 

PROCEDURE settables(v1(mV)) {

	TABLE minf, hinf, mtau, htau  FROM -120 TO 40 WITH 641

	minf  = 1 / ( 1 + exp( ( - ( v1 + fastNa_shift ) - 38 ) / 10 ) )
	if( ( v1 + fastNa_shift ) < -30.0 ) {
		mtau = 0.025 + 0.14 * exp( ( ( v1 + fastNa_shift ) + 30 ) / 10 )
	} else{
		mtau = 0.02 + a + (0.145+ b) * exp( ( - ( v1 + fastNa_shift +d ) - 30 ) / (10+c) ) 
	}

	: hinf, and htau are shifted 3.5 mV comparing to the paper

	hinf  = 1 / ( 1 + exp( ( ( v1 + fastNa_shift * 0 ) + 62.9 ) / 10.7 ) )
	htau = 0.15 + 1.15 / ( 1 + exp( ( ( v1 + fastNa_shift * 0 ) + 37 ) / 15 ) )
}

UNITSON
