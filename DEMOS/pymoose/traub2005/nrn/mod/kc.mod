TITLE Potasium C type current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
	(mM) = (milli/liter)
}
 
NEURON { 
	SUFFIX kc
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE  gbar, ik, alpha, beta, iktmp, df, gamma
}

PARAMETER { 
	gbar = 0.0 	(mho/cm2)
	v (mV) ek 		(mV)  
	cai		(mM)
} 

ASSIGNED { 
	ik 		(mA/cm2) 
	iktmp 		(mA/mho) 
	alpha (/ms) beta	(/ms)
	df		(mV)
	gamma		(mM)
}
 
STATE {
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	if( 0.004(1/mM) * cai < 1 ) {
		iktmp =  m * 0.004(1/mM) * cai * ( v - ek ) 
		gamma = 0.004 * cai
	}else{
		iktmp =  m * ( v - ek )
	        gamma = 1.0
	}
	df = v-ek
	ik = gbar*iktmp
}
 
INITIAL { 
	settables(v) 
	m = alpha / ( alpha + beta )
	m = 0
}
 
DERIVATIVE states { 
	settables(v) 
	m' = alpha * ( 1 - m ) - beta * m 
}

UNITSOFF 

PROCEDURE settables(v (mV)) { 
	TABLE alpha, beta FROM -120 TO 40 WITH 641

	if( v < -10.0 ) {
		alpha = 2 / 37.95 * ( exp( ( v + 50 ) / 11 - ( v + 53.5 ) / 27 ) )

		: Note that there is typo in the paper - missing minus sign in the front of 'v'
		beta  = 2 * exp( ( - v - 53.5 ) / 27 ) - alpha
	}else{
		alpha = 2 * exp( ( - v - 53.5 ) / 27 )
		beta  = 0
	}
}

UNITSON
