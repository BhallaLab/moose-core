TITLE Borg-Graham type generic K-A channel
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
	v (mV)
        ek (mV)
	celsius 	(degC)
	gkabar=.01 (mho/cm2)
        vhalfn=-33.6   (mV)
        vhalfl=-83   (mV)
        a0l=0.08      (/ms)
        a0n=0.02    (/ms)
        zetan=-3    (1)
        zetal=4    (1)
        gmn=0.6   (1)
        gml=1   (1)
}


NEURON {
	SUFFIX borgka
	USEION k READ ek WRITE ik
        RANGE gkabar,gka
        GLOBAL ninf,linf,taul,taun
}

STATE {
	n
        l
}

INITIAL {
        rates(v)
        n=ninf
        l=linf
}

ASSIGNED {
	ik (mA/cm2)
        ninf
        linf      
        taul
        taun
        gka
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gka = gkabar*n*l
	ik = gka*(v-ek)

}


FUNCTION alpn(v(mV)) {
  alpn = exp(1.e-3*zetan*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betn(v(mV)) {
  betn = exp(1.e-3*zetan*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION alpl(v(mV)) {
  alpl = exp(1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betl(v(mV)) {
  betl = exp(1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
}

DERIVATIVE states { 
        rates(v)
        n' = (ninf - n)/taun
        l' = (linf - l)/taul
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a,q10
        q10=3^((celsius-30)/10)
        a = alpn(v)
        ninf = 1/(1 + a)
        taun = betn(v)/(q10*a0n*(1+a))
        a = alpl(v)
        linf = 1/(1+ a)
        taul = betl(v)/(q10*a0l*(1 + a))
}

