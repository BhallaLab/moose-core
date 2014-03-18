# This example illustrates loading and running, a kinetic model 
# defined in cspace format. We use the gsl solver here. The model already
# defines a couple of plots and sets the runtime to 100 seconds. The
# script dumps the output into an xplot file called data.plot and the
# saved version into saveReaction.g

import math
import pylab
import numpy
import moose
def main():
		# This command loads the file into the path '/model', and tells
		# the system to use the gsl solver.
		modelId = moose.loadModel( 'Osc.cspace', 'model', 'Neutral' )
		moose.useClock( 4, "/model/kinetics/##[]", "process" )
		moose.setClock( 4, 0.01 )
		moose.setClock( 8, 1 )
		"""
		moose.useClock( 5, "/model/graphs/##", "process" )
		moose.setClock( 5, 1 )
		"""
		moose.reinit()
		moose.start( 3000.0 ) # Run the model for 300 seconds.

		# display all plots
		for x in moose.wildcardFind( '/model/graphs/#' ):
			t = numpy.arange( 0, x.vector.size, 1 ) #sec
			pylab.plot( t, x.vector, label=x.name )
		pylab.legend()
		pylab.show()

		# moose.saveModel( modelId, 'saveReaction.g' )
		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
