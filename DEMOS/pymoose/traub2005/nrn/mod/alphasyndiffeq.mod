: alphasyndiffeq.mod is actually
: exp2syn.mod (default supplied with NEURON) modified so that the
: time constants are very close to each other.  The new global
: near_unity_AlphaSynDiffEq is the factor multiplied into
: tau2 to make tau1.
COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak conductance is 1.
Decay time, tau2, MUST be greater than rise time, tau1.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS AlphaSynDiffEq
	RANGE tau1, tau2, e, i
	NONSPECIFIC_CURRENT i

	RANGE g
	GLOBAL total, near_unity
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	near_unity = 0.999 (1) : tau1 tenth of a percent smaller than tau2 by default
	tau2 = 10 (ms) <1e-9,1e9>
	e=0	(mV)
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	total (uS)
	tau1 (ms)
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	total = 0
	tau1 = near_unity * tau2
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight (uS)) {
	state_discontinuity(A, A + weight*factor)
	state_discontinuity(B, B + weight*factor)
	total = total+weight
}
