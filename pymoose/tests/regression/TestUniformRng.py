from math import *
from numpy import *

from moose import *

def testUniformRng(tolerance=1e-3, maxSamples=1000):
    tests = Neutral("/tests")
    tables = Neutral("testResults")
    rng = UniformRng("uniformRng", tests)
    data = Table("uniformRng", tables)
    data.connect("inputRequest", rng, "sample")
    rng.useClock(0)
    data.useClock(0)
    tests.getContext().step(maxSamples)
    data.dumpFile("uniform_rng.plot")
    # compare sample mean and variance with theoretical values
    sample_mean = [sum(sample) for sample in data] / len(data)
    if fabs(sample_mean - rng.mean) > tolerance*fabs(rng.mean):
	print "FAILED:", "sample_mean =", sample_mean, ", mean =", mean
    sample_var = [sum((sample - sample_mean) * (sample - sample_mean)) for sample in data] / len(data)
    if fabs(sample_var - rng.var) > tolerance*fabs(rng.var):
	print "FAILED:", "sample_mean =", sample_mean, ", mean =", mean
