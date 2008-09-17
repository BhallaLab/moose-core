from math import *

import moose

from testutil import *

allowable_relative_error = 0.01

class TestNormalRNG(moose.Neutral):
	"""Test random normal distribution generator"""
	def __init__(self, *args):
		moose.Neutral.__init__(self, *args)
		self.testCount = 0
	
	def run_single(self, mean_=0.0, variance_=1.0, steps=1000):
		self.rng = moose.NormalRng("normalRNG" + str(self.testCount), self)
		self.rng.method = 2
		self.table = moose.Table("normalRNGTable" + str(self.testCount), self)
		self.table.stepMode = 3
		self.table.connect("inputRequest", self.rng, "sample")
		self.getContext().useClock(0, self.path()+"/##")
		self.getContext().reset()
# 		self.table.useClock(0)
# 		self.rng.useClock(0)
		self.testCount += 1
		self.rng.mean = mean_
		if not is_equal(self.rng.mean, mean_):
			raise Exception("Error setting mean: tried:" + mean_ + "actual:" + self.rng.mean)
		self.rng.variance = variance_
		if not is_equal(self.rng.variance, variance_):
			raise Exception("Error setting variance: tried:" + variance_ + "actual:" + self.rng.variance)
		self.getContext().reset()
	
		self.getContext().step(int(steps))
		print "Running for", steps, "steps. Table length", len(self.table)
		tmp_mean = sum(value for value in self.table) * 1.0 / steps
		tmp_var = sum( (tmp_mean - value) ** 2 for value in self.table ) / steps
		
# 		# Var(x) = Integral( p(x) * ( x - u ) ** 2 ) wrt x, where u is the mean, p(x) is probability density at x
# 		# For normal distribution: p(x) = exp( - (x - u) ** 2 / ( 2 * var **2 )) / (var * sqrt( 2 * pi))
# 		for value in self.table:
# 			norm = (mean - value) / variance
# 			density = exp( -0.5 * norm * norm) / (variance * sqrt(2 * pi)) 
# 			tmp_var +=  (mean - value) * density
		return (tmp_mean, tmp_var)

	def run_test(self, mean=0.0, var=1.0, steps=1000, count = 1000):
		print "run_test(mean=", mean, ", var=", var, ")"
		result = []
		for ii in range(count):
			result.append(self.run_single(mean, var, steps))
		xmean = sum(x[0] for x in result) / count
		xvar = sum(x[1] * x[1] for x in result) / count
		if fabs(xmean - mean) > allowable_relative_error:
			print "ERROR: actual mean deviates by:", fabs(xmean - mean), ": xmean =", xmean, ": mean =", self.rng.mean
			return False
		if fabs(xvar - var) > allowable_relative_error:
			print "ERROR: actual variance deviates by:", fabs(xvar - var), ": xvar =", xvar, ": var =", self.rng.variance 
			return False
		return True

	def save_plot(self, file_name="normal_rng.plot"):
		self.table.dumpFile(file_name)		
	
if __name__ == "__main__":
	test_object = TestNormalRNG("testNormalRNG")
	print "asked: (1.0e-7, 0.3e-7)", ", got:", test_object.run_single(0.0, 1e-8)
	test_object.save_plot()
	
