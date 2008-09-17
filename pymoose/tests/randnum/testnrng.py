from math import *

from pylab import *

import moose

nrng = moose.NormalRng("nrng")
nrng.mean = 1.0
nrng.variance = 3.0
nrng_table = moose.Table("nrng_table")
nrng_table.stepMode = 3
nrng_table.connect("inputRequest", nrng, "sample")
nrng.useClock(0)
nrng_table.useClock(0)
nrng.getContext().reset()
nrng.getContext().step(1000)
print "mean:", mean(array(nrng_table)), "variance:", var(array(nrng_table))
