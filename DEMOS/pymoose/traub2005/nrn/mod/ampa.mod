COMMENT
 ampa.mod is 
 alphasyndiffeqt.mod which is actually
 exp2syn.mod (default supplied with NEURON) modified so that the
 time constants are very close to each other.  The new global
 near_unity_AlphaSynDiffEqT is the factor multiplied into
 tau2 to make tau1. 
 Note: that tau2 was renamed tau so that it would be obvious
 which time constant to set.
This program was then further modified to make
 more similar to Traub et al 2005:
delta = time-presyn
dexparg = delta/tau
if (dexparg <= 100
	z = exp(-dexparg)
else
	z = 0
endif
g = g + g_0 * delta * z
 
and current = (g_ampa + open(i) * g_nmda) * V - g_gaba_a (V-V_gaba_a)
i.e. the reversal potential for ampa and nmda is 0.

Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak conductance is 1.
Decay time, tau2, MUST be greater than rise time, tau1.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor used to be evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1, however now it is set so that a peak
conductance of tau2*exp(-1) is reached because that's what the
Traub alpha function (t-t_0)*exp(-(t-t_0)/tau) reaches..

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS AMPA  : since only used for ampa, a preferable name to AlphaSynDiffEqT
	RANGE tau, e, i : tau1 removed from RANGE because under program cntrl
			: what was tau2 was renamed tau for easy remembering
			: during use of this synapse
	NONSPECIFIC_CURRENT i

	RANGE g
	GLOBAL total, near_unity, gfac

:for network debugging 
:	USEION ampa1 WRITE iampa1 VALENCE 0
:	USEION ampa2 WRITE iampa2 VALENCE 0
:	RANGE srcgid, targid, comp, synid
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	near_unity = 0.999 (1) : tau1 tenth of a percent smaller than tau2 by default
	tau = 10 (ms) <1e-9,1e9>
	e=0	(mV)
	gfac = 1
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	total (uS)
	tau1 (ms)

:	iampa1 (nA)
:	iampa2 (nA)
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
	LOCAL tp
	total = 0
	tau1 = near_unity * tau
	A = 0
	B = 0
	tp = (tau1*tau)/(tau - tau1) * log(tau/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau)
	factor = 1/factor
:	The above factor gives a peak conductance of 1
:	The above code is kept in place for comparison
:	This is modified though to return a peak value of tau*exp(-1)
:	(see FORTRAN code: f_traub = (t-t_0)*exp(-(t-t_0)/tau))
	factor = factor * tau * exp(-1)*1(/ms)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	g = gfac*g
	i = g*(v - e)
:	iampa1 = g
:	iampa2 = -g
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau
}

NET_RECEIVE(weight (uS)) {
	A = A + weight*factor
	B = B + weight*factor
	total = total+weight
}
