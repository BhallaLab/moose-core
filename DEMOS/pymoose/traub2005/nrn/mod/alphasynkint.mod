COMMENT
alphasynkint.mod
Alpha Synapse Traub-like implemented with Kinetic Scheme as per 
Chapter 10 NEURON book
Used to return peak conductance of 1, however now it is set so that 
a peak conductance of tau2*exp(-1) is reached because that's what
the Traub alpha function (t-t_0)*exp(-(t-t_0)/tau) reaches..
ENDCOMMENT
NEURON {
	POINT_PROCESS AlphaSynKinT : ending T is for Traub, see notes
	RANGE tau, e, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE { a (microsiemens) g (uS) }

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD sparse
	i = g*(v - e)
}

KINETIC state {
	~ a <-> g (1/tau, 0)
	~ g -> (1/tau)
}

NET_RECEIVE(weight (uS)) {
:	a = a + weight*exp(1) * (tau*exp(-1))
: the above last factor changes peak conductance to from
: 1 to tau*exp(-1) so formula becomes:
	a = a + weight*tau*1(/ms)
}
