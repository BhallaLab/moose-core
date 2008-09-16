import types

from moose import *

rng = ["Binomial", "Exponential", "Gamma", "Normal", "Poisson" ]

mooseClasses = []
class MooseClasses:
	def __init__(self):
		self._classList = []
		class_file = open("classes.txt", "r")
		for class_name in class_file:
			class_name = class_name.strip()
			if len(class_name) > 0:
				classObj = eval(class_name)
				self._classList.append(classObj)
			
	def classList(self):
		"""Returns a list of MOOSE classes available in PyMOOSE"""
		return self._classList

	def testCreation(self):
		"""Try to create each class in list"""
		result = True
		container = Neutral("/testContainer")
		for mooseClass in self._classList:
			print "Create", mooseClass.__name__,
			entity = mooseClass(mooseClass.__name__,container)
			if entity.id.good():
				print "- OK"
			else:
				print "- Failed"
				result = False
		return result

if __name__ == "__main__":
	test = MooseClasses()
	test.testCreation()


