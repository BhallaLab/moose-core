from math import *
from numpy import *

from moose import *
from kstest import *

_min = 0.0
_max = 1.0

def uniform_distrfn(x): # we cannot pass min and max as params because the ks test takes a single valued fn.
    assert((_min <= x) and (_max > x))
    return x/(_max - _min)


# This test does not pass - the KS-test failed 34 times out of 200 runs in sequence
def full_rng_test(testNo=0, tolerance=0.1, maxSamples=1000):
    result = True
    tests = Neutral("/tests")
    tables = Neutral("testResults")
    # First test the default setting: min = 0.0, max = 1.0, mean = 0.5
    rng = UniformRng("uniformRng" + str(testNo), tests)
    data = Table("uniformRng" + str(testNo), tables)
    data.stepMode = 3
    data.connect("inputRequest", rng, "sample")
    rng.useClock(0)
    data.useClock(0)
    tests.getContext().reset()
    tests.getContext().step(maxSamples-1) # step goes 1 step extra
    data.dumpFile("uniform_rng.plot")
    sample = array(data)
    # compare sample mean and variance with theoretical values
    sample_mean = sample.mean()
    if fabs(sample_mean - rng.mean) > tolerance*fabs(rng.mean):
	print "FAILED:", "sample_mean =", sample_mean, ", intended mean =", rng.mean
        result = False
    sample_var = sum((sample - sample_mean) ** 2 ) / len(data)
    if fabs(sample_var - rng.variance) > tolerance*fabs(rng.variance): # note: this fails when mean = 0.0
	print "FAILED:", "sample_variance =", sample_var, ", intended variance =", rng.variance
        result = False
    # do a max of 5 test
    if maxSamples % 5 != 0:
        maxSamples = 5 * (maxSamples/5)
        sample = sample[:maxSamples]
    sample.resize(maxSamples/5, 5)
    maxes = apply_along_axis(max, 1, sample)
    result = ks_test(maxes, lambda x: x**5)
    return result

def do_full_rng_test(count=100, tolerance=0.1, maxSample=1000):
    """Run full_rng_test count times with tolerance and maxSample number of samples"""
    result = 0
    for ii in range(count):
        if full_rng_test(ii, tolerance, maxSample):
            result += 1
    return result

def testUniformRng(testId, min=0.0, max=1.0, tolerance = 0.1, sampleCount=1000):
    result = True
    tests = Neutral("/tests")
    tables = Neutral("/testResults")
    # First test the default setting: min = 0.0, max = 1.0, mean = 0.5
    rng = UniformRng("uniformRng" + str(testId), tests)
    rng.min = min
    if fabs(rng.min - min) > finfo(float).eps:
	print "FAILED:", testId, ":: actual lower bound =", rng.min, ", intended lower bound =", min
        result = False
    rng.max = max
    if fabs(rng.max - max) > finfo(float).eps:
	print "FAILED:", testId, ":: actual upper bound =", rng.max, ", intended upper bound =", max
        result = False
    data = Table("uniformRng" + str(testId), tables)
    data.stepMode = 3
    data.connect("inputRequest", rng, "sample")
    rng.useClock(0)
    data.useClock(0)
    tests.getContext().reset()
    tests.getContext().step(sampleCount-1) # step goes 1 step extra
    sample = array(data)
    # compare sample mean and variance with theoretical values
    sample_mean = sample.mean()
    if fabs(sample_mean - rng.mean) > tolerance*fabs(rng.mean):
	print "FAILED:", testId, ":: sample_mean =", sample_mean, ", intended mean =", rng.mean
        result = False
    sample_var = sum((sample - sample_mean) ** 2 ) / len(data)
    if fabs(sample_var - rng.variance) > tolerance*fabs(rng.variance): # note: this fails when mean = 0.0
	print "FAILED:", testId, ":: sample_variance =", sample_var, ", intended variance =", rng.variance
        result = False
    return result

def doTest():
    return testUniformRng("urng1") and \
        testUniformRng("urng2", 1.0, 10.0) and \
        testUniformRng("urng3", 1e5, 1e12) and \
        testUniformRng("urng4", -10.0, -1.0) and \
        testUniformRng("urng5", -1e12, -1e5) and \
        testUniformRng("urng6", -1.0, 1.0) # this test fails, mean close to 0.0 goes off in samples for some reason
        
if __name__ == "__main__":
    print "Testing UnifromRng: passed?", doTest()
