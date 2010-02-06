NEURON {
	POINT_PROCESS GABAA
	RANGE tau, e, i
	NONSPECIFIC_CURRENT i
	GLOBAL gfac
: for network debugging
:	USEION gaba1 WRITE igaba1 VALENCE 0
:	USEION gaba2 WRITE igaba2 VALENCE 0
:	RANGE srcgid, targid, comp, synid
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
	gfac = 1
}

ASSIGNED {
	v (mV)
	i (nA)
:	igaba1 (nA)
:	igaba2 (nA)
:	srcgid
:	targid
:	comp
:	synid
}

STATE {
	g (uS)
}

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = gfac*g*(v - e)
:	igaba1 = g
:	igaba2 = -g
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(weight (uS)) {
	state_discontinuity(g, g + weight)
}
