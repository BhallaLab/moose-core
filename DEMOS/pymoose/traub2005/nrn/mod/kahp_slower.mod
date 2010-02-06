TITLE Potasium AHP (slower) type current for RD Traub, et al 2005

COMMENT
	Modified by Tom Morse to make slower than the slow suppyrFRB, suppyrRS AHP.
	Note the time constant at zero cai is 1 second here and 100ms in the slow AHP
	3/13/06
	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)
	RD Traub, J Neurophysiol 89:909-921, 2003
ENDCOMMENT

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
	(mM) = (milli/liter)
} 

NEURON { 
	SUFFIX kahp_slower
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gbar, ik, m
}

PARAMETER { 
	gbar = 0.0 	(mho/cm2)
	v		(mV) 
	ek 		(mV)  
	cai		(mM)
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
	rates( cai )
	m = alpha / ( alpha + beta )
	m = 0
}
 
DERIVATIVE states { 
	rates( cai )
	m' = alpha * ( 1 - m ) - beta * m 
}

UNITSOFF 

PROCEDURE rates(chi (mM)) { 

	if( cai < 500 ) {
		alpha = cai / 50000
	}else{
		alpha = 0.01
	}
	beta = 0.001
}

UNITSON
