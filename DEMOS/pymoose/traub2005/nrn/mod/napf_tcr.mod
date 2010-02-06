TITLE Sodium persistent current for RD Traub et al 2003, 2005

COMMENT

	This persistent sodium current is based on the activation
	permissive quantity, m, from the transient sodium channel. -TMM
	modified from an
	Implementation by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)
	fastNashift init to 0 and removed from arg modification Tom Morse 3/8/2006
	(for Traub et al 2005)
	The difference between napf and napf_tcr is that napf_tcr has a single power
	of m in ina_napf_tcr where as napf has the third power of m in ina_napf
ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX napf_tcr
	USEION na READ ena WRITE ina
	RANGE gbar, ina,m, df, fastNa_shift, a, b, c, d, minf, mtau
}
PARAMETER { 
	fastNa_shift = 7 (mV)
	a = 0 (1)
	b = 0 (1)
	c = 0 (1)
	d = 0 (1)
	gbar = 0.0 	   (mho/cm2)
	v (mV) ena 		   (mV)  
} 
ASSIGNED { 
	ina 		   (mA/cm2) 
	minf 	   (1)
	mtau 	   (ms)
	df	(mV)
} 
STATE {
	m
}
BREAKPOINT { 
	SOLVE states METHOD cnexp
	ina = gbar * m * ( v - ena ) 
	df = v - ena
} 
INITIAL { 
	settables( v )
	m = minf
	m = 0
} 
DERIVATIVE states { 
	settables( v ) 
	m' = ( minf - m ) / mtau 
}

UNITSOFF 

PROCEDURE settables(v1(mV)) {

	TABLE minf, mtau  FROM -120 TO 40 WITH 641

	minf  = 1 / ( 1 + exp( ( - ( v1 + fastNa_shift ) - 38 ) / 10 ) )
	if( ( v1 + fastNa_shift ) < -30.0 ) {
		mtau = 0.025 + 0.14 * exp( ( ( v1 + fastNa_shift ) + 30 ) / 10 )
	} else {
		mtau = 0.02 + a + (0.145+ b) * exp( ( - ( v1 + fastNa_shift +d ) - 30 ) / (10+c) ) 
	}

}

UNITSON
