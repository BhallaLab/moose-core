: alphasynkin.mod
: Alpha Synapse implemented with Kinetic Scheme as per Chapter 10 NEURON book
NEURON {
	POINT_PROCESS AlphaSynKin
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
	a = a + weight*exp(1)
}
