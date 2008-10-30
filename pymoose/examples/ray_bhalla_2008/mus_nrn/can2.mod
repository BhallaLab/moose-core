TITLE n-calcium channel
: n-type calcium channel


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

	FARADAY = 96520 (coul)
	R = 8.3134 (joule/degC)
	KTOMV = .0853 (mV/degC)
}

PARAMETER {
	v (mV)
	celsius 		(degC)
	gcanbar=.0003 (mho/cm2)
	ki=.001 (mM)
	cai=5.e-5 (mM)
	cao = 10  (mM)
}


NEURON {
	SUFFIX can
	USEION ca READ cai,cao WRITE ica
        RANGE gcanbar       
        GLOBAL hinf,minf,taum,tauh
}

STATE {
	m h 
}

ASSIGNED {
	ica (mA/cm2)
        gcan  (mho/cm2) 
        minf
        hinf
        taum
        tauh
}

INITIAL {
        rates(v)
        m = minf
        h = hinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gcan = gcanbar*m*m*h*h2(cai)
	ica = gcan*ghk(v,cai,cao)

}

UNITSOFF
FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}


FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f

        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {
        KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}

FUNCTION alph(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alph = 1.6e-4*exp(-v/48.4)
}

FUNCTION beth(v(mV)) {
        TABLE FROM -150 TO 150 WITH 200
	beth = 1/(exp((-v+39.0)/10.)+1.)
}

FUNCTION alpm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alpm = 0.1967*(-1.0*v+19.88)/(exp((-1.0*v+19.88)/10.0)-1.0)
}

FUNCTION betm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	betm = 0.046*exp(-v/20.73)
}

UNITSON

DERIVATIVE states {     : exact when v held constant; integrates over dt step
        rates(v)
        m' = (minf - m)/taum
        h' = (hinf - h)/tauh
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a
        a = alpm(v)
        taum = 1/(a + betm(v))
        minf = a*taum
        a = alph(v)
        tauh = 1/(a + beth(v))
        hinf = a*tauh
}











