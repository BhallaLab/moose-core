TITLE Potasium M type current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
}
 
NEURON { 
	SUFFIX km
	USEION k READ ek WRITE ik
	RANGE gbar, ik
}

PARAMETER { 
	gbar = 0.0 	(mho/cm2)
	v (mV) ek 		(mV)  
}
 
ASSIGNED { 
	ik 		(mA/cm2) 
	alpha (/ms) beta	(/ms)
}
 
STATE {
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	ik = gbar * m * ( v - ek ) 
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
	alpha = 0.02 / ( 1 + exp( ( -v - 20 ) / 5 ) )
	beta  = 0.01 * exp( ( -v - 43 ) / 18 )
}

