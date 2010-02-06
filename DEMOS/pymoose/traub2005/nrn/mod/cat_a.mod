TITLE Calcium low threshold T type current version a for RD Traub 2005

COMMENT
	This current is found in the model deepLTS cells
	Traub made this model modifying Huguenard and Prince 1992 as
	appeared on Destexhe et al 1996 (in vivo in vitro ...)
	Modification by Tom Morse for Traub et al 2005
	modified from
	Implementation by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)
	RD Traub, J Neurophysiol 89:909-921, 2003
ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
}
 
NEURON { 
	SUFFIX cat_a
	NONSPECIFIC_CURRENT i   : not causing [Ca2+] influx
	RANGE gbar, i, m, h, alphah, betah 	: m,h, alphah, betah for comparison with FORTRAN
}

PARAMETER { 
	gbar = 0.0 	(mho/cm2)
	v 		(mV)  
}
 
ASSIGNED { 
	i 		(mA/cm2) 
	minf hinf 	(1)
	mtau (ms) htau 	(ms) 
	alphah (/ms) betah	(/ms)
}
 
STATE {
	m h
}

BREAKPOINT { 
	SOLVE states METHOD cnexp 
	i = gbar * m * m * h * ( v - 125 ) 
	alphah = hinf/htau
	betah = 1/htau - alphah
}
 
INITIAL { 
	settables(v) 
:	m  = minf
	h  = hinf
	m  = 0
} 

DERIVATIVE states { 
	settables(v) 
	m' = ( minf - m ) / mtau 
	h' = ( hinf - h ) / htau
}

UNITSOFF 

PROCEDURE settables(v(mV)) { 
	TABLE minf, mtau,hinf, htau FROM -120 TO 40 WITH 641
        minf  = 1 / ( 1 + exp( ( -v - 52 ) / 7.4 ) )
        mtau  = 1 + .33 / ( exp( ( v + 27.0 ) / 10.0 ) + exp( ( - v - 102 ) / 15.0 ) )

        hinf  = 1 / ( 1 + exp( ( v + 80 ) / 5 ) )
        htau = 28.30 +.33 / (exp(( v + 48.0)/ 4.0) + exp( ( -v - 407.0) / 50.0 ) )

}

UNITSON
