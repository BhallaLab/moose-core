import moose
import sys
import math
import pylab
import numpy

moose.loadModel('/home/harsha/async/Demos/snippets/replicasbml11.g','/bc')

#moose.setClock(4,0.1)
#moose.useClock(4,"/sbml1/compartment/##[]","process")
#moose.setClock(8,0.1)
#moose.useClock(8,"/model/graphs/#","process")

moose.reinit()
moose.start(20)

for x in moose.wildcardFind( '/bc/graphs/##[ISA=Table]' ):
	t = numpy.arange( 0, x.vector.size, 1 ) # sec
	pylab.plot( t, x.vector, label=x.name )
pylab.legend()
pylab.show()


moose.loadModel('/home/harsha/async/Demos/snippets/reacwithouts1.g','/bc1')
moose.reinit()
moose.start(20)
for x in moose.wildcardFind( '/bc1/graphs/##[ISA=Table]' ):
	t = numpy.arange( 0, x.vector.size, 1 ) # sec
	pylab.plot( t, x.vector, label=x.name )
pylab.legend()
pylab.show()

quit()

