COMMENT
traub_nmda.mod
Traub-like NMDA synaptic current
This file is a merge of rampsyn.mod and expsyn.mod
The Traub et al 2005 paper contains a nmda synaptic current which
when activated has a linear ramp (in conductance) up to the conductance scale
over 5ms, then there is an exponential decay (in conductance).
Tom Morse, Michael Hines
ENDCOMMENT
NEURON {
	POINT_PROCESS NMDA
	RANGE tau, time_interval, e, i,weight, NMDA_saturation_fact, flag, g
	NONSPECIFIC_CURRENT i
	GLOBAL gfac
: for network debugging
:	USEION nmda1 WRITE inmda1 VALENCE 0
:	USEION nmda2 WRITE inmda2 VALENCE 0
:	RANGE srcgid, targid, comp, synid
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}

PARAMETER {
	tau = 130.5 (ms)  <1e-9,1e9>	: NMDA conductance decay time constant
: default choice is tauNMDA_suppyrRS_to_suppyrRS=130.5e0, a sample tau from groucho.f
	time_interval = 5 (ms) <1e-9,1e9>
	e = 0	(mV)
	weight = 2.5e-8 (uS)	: example conductance scale from Traub 2005 et al
			 	: gNMDA_suppyrRS_to_suppyrRS (double check units)
	NMDA_saturation_fact=1e10 (1) : 80e0 (1) : this saturation factor is multiplied into
		: the conductance scale, weight, for testing against the
		: instantaneous conductance, to see if it should be limited.
: FORTRAN nmda subroutine constants and variables here end with underbar 
	A_ = 0 (1) : initialized with below in INITIAL, assigned in each integrate_celltype.f
	BB1_ = 0 (1) : assigned in each integrate_celltype.f
	BB2_ = 0 (1) : assigned in each integrate_celltype.f
	Mg = 1.5 (mM) : a FORTRAN variable set in groucho.f
	gfac = 1
}

ASSIGNED {
	v (mV)
	i (nA)
	event_count (1)	: counts number of syn events being processed
	k (uS/ms) : slope of ramp or 0
	g (uS)
	A1_ (1)
	A2_ (1)
	B1_ (1)
	B2_ (1)
	Mg_unblocked (1)
:	inmda1 (nA)
:	inmda2 (nA)
:	srcgid
:	targid
:	comp
:	synid
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	A_ =  exp(-2.847)  : assigned in each integrate_celltype.f
	BB1_ = exp(-.693)  : assigned in each integrate_celltype.f
	BB2_ = exp(-3.101) : assigned in each integrate_celltype.f
	g = 0
	A = 0
	B = 0
	k = 0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = A + B
	if (g > NMDA_saturation_fact * weight) { g = NMDA_saturation_fact * weight }
	g = g*gfac
	i = g*Mg_unblocked*(v - e)
:	inmda1 = g
:	inmda2 = -g
}

DERIVATIVE state {
	Mg_factor()
	B' = -B/tau
	A' = k
}

NET_RECEIVE(weight (uS)) {
	if (flag>=1) {
		: self event arrived, terminate ramp up
	: remove one event's contribution to the slope, k
		k = k - weight/time_interval
	: Transfer the conductance over from A to B
		B = B + weight
		A = A - weight
	} else {
		: stimulus arrived, make or continue ramp
		net_send(time_interval, 1) : self event to terminate ramp
	: add one event ramp to slope k:
		k = k + weight/time_interval
:	note there are no state discontinuities at event start since the begining of a ramp
:	only has a discontinuous change in derivative
	}
}

: an NMDA subroutine converted from FORTRAN whose sole purpose was to compute the number
: of open nmda recpt channels due to relief from Mg block

PROCEDURE Mg_factor() {
UNITSOFF
           A1_ = exp(-.016*v - 2.91)
           A2_ = 1000.0 * Mg * exp (-.045 * v - 6.97)
           B1_ = exp(.009*v + 1.22)
           B2_ = exp(.017*v + 0.96)
UNITSON
           Mg_unblocked  = 1.0/(1.0 + (A1_+A2_)*(A1_*BB1_ + A2_*BB2_) /
                 (A_*A1_*(B1_+BB1_) + A_*A2_*(B2_+BB2_))  )
}
