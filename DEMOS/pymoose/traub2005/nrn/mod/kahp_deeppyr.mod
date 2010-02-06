TITLE Potasium AHP type current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)

ENDCOMMENT

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
	(mM) = (milli/liter)
} 

NEURON { 
	SUFFIX kahp_deeppyr
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

	if( cai < 100 ) {
		alpha = cai * 1e-4
	}else{
		alpha = 0.01
	}
	beta = 0.001  : 0.01 for ordinary kahp
}

UNITSON
