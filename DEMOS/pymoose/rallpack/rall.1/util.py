import moose
import math
context = moose.PyMooseBase.getContext()

def make_compartment(path, RA, RM, CM, EM, inject, diameter, length):
	Ra = 4.0 * length * RA / ( math.pi * diameter * diameter )
	Rm = RM / ( math.pi * diameter * length )
	Cm = CM * math.pi * diameter * length

	compartment = moose.Compartment(path)
	compartment.Ra = Ra
	compartment.Rm = Rm
	compartment.Cm = Cm
	compartment.Em = EM
	compartment.inject = inject
	compartment.length = length
	compartment.diameter = diameter
	compartment.initVm = EM

def link_compartment(path1, path2):
	context.connect(context.pathToId(path1), 'raxial', context.pathToId(path2), 'axial')
