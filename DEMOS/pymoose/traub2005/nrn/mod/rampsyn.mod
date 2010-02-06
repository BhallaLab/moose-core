COMMENT
rampsyn.mod
for use with expsyn.mod to make a Traub-like NMDA receptor
Tom Morse, Michael Hines
ENDCOMMENT
NEURON {
	POINT_PROCESS RampSyn
	RANGE time_interval, e, i, weight, saturation_fact, k
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	time_interval = 5 (ms) <1e-9,1e9>
	e = 0	(mV)
	weight = 2.5e-5 (uS)	: example conductance scale from Traub 2005 et al
			 	: gNMDA_suppyrRS_to_suppyrRS (double check units)
	saturation_fact=1e10 (1) :80e0 (1) : this saturation factor is multiplied into
		: the conductance scale, weight, for testing against the
		: instantaneous conductance, to see if it should be limited.
}

ASSIGNED {
	v (mV)
	i (nA)
	k (uS/ms)
}

STATE {
	g (uS)
}

INITIAL {
	g=0
	k=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	if (g > saturation_fact * weight) { g = saturation_fact * weight }
	i = g*(v - e)
}

DERIVATIVE state {
	g' = k
}

NET_RECEIVE(weight (uS)) {
	if (flag>=1) {
		: self event arrived, terminate ramp
		k = k - weight/time_interval
		g = g - weight
	} else {
		: stimulus arrived, make or continue ramp
		net_send(time_interval, 1) : self event to terminate ramp
		k = k + weight/time_interval
	}
}
