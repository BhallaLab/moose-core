from matplotlib import pyplot as plt
import numpy

CMg = 2.0                     # [Mg] in mM
eta = 0.33                    # per mM
gamma = 60                    # per Volt

A = 1.0 / eta
B = 1.0 / gamma

v_min = -100e-3
v_max = 100e-3
v_n = 200

def fraction( v ):
	return ( A / ( A + CMg * numpy.exp( -v / B ) ) )

v = numpy.linspace( v_min, v_max, v_n )
f = fraction( v )

plt.plot( v, f )
plt.savefig( 'blocked_fraction.png' )
