TITLE Borg-Graham type generic K-AHP channel

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
	celsius 	(degC)
	gkahpbar=.003 (mho/cm2)
        n=4
        cai=50.e-6  (mM)
        a0=1.3e13	(/ms-mM-mM-mM-mM)	:b0/(1.4e-4^4)
        b0=.5e-2 	(/ms)			:0.5/(0.100e3)
        v       (mV)
        ek      (mV)
}


NEURON {
	SUFFIX kahp
	USEION k READ ek WRITE ik
        USEION ca READ cai
        RANGE gkahpbar,gkahp
        GLOBAL inf,tau
}

STATE {
	w
}

ASSIGNED {
	ik (mA/cm2)
        gkahp  (mho/cm2)
        inf
        tau
}

INITIAL {
        rate(cai)
        w=inf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gkahp = gkahpbar*w
	ik = gkahp*(v-ek)

}

FUNCTION alp(cai (mM)) {
  alp = a0*cai^n
}

DERIVATIVE state {     : exact when v held constant; integrates over dt step
        rate(cai)
        w' = (inf - w)/tau
}

PROCEDURE rate(cai (mM)) { :callable from hoc
        LOCAL a
        a = alp(cai)
        tau = 1/(a + b0)
        inf = a*tau
}















