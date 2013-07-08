# This is a toy model of synaptic bidirectional plasticity. The model has 
# a small a bistable chemical switch, and a small set of reactions that
# decode calcium input. One can turn the switch on with short high 
# calcium pulses (over 2 uM for about 10 sec). One can turn it back off
# again using a long, lower calcium pulse (0.2 uM, 1000 sec).
# The script dumps the output into an xplot file called 
# bidirectionalPlasticity.plot.

import moose
import os
import pylab

def dumpPlots():
		fname = 'bidirectionalPlasticity.plot'
		if ( os.path.exists( fname ) ):
				os.remove( fname )
		for x in moose.wildcardFind( '/model/graphs/conc#/#,/model/moregraphs/conc#/#' ):
			moose.element( x[0] ).xplot( fname, x[0].name )

def displayPlots():
		clock = moose.Clock( '/clock' ) # look up global clock
		totR = moose.element( '/model/graphs/conc1/tot_PSD_R.Co' )
		PP1 = moose.element( '/model/moregraphs/conc4/PP1_active.Co' )
		Ca = moose.element( '/model/graphs/conc1/Ca.Co' )
		pylab.plot( pylab.linspace( 0, clock.currentTime, len( totR.vec )), totR.vec, label='membrane Receptor' )
		pylab.plot( pylab.linspace( 0, clock.currentTime, len( PP1.vec ) ), PP1.vec, label='active PP1' )
		pylab.plot( pylab.linspace( 0, clock.currentTime, len( Ca.vec ) ), Ca.vec, label='Ca' )
		pylab.legend()
		pylab.show()

def main():
		# Pick your favourite numerical method.
		method = 'gsl'
		#gssa turns off only about 20% of the time, perhaps a longer turnoff
		# stimulus time would increase this.
		#method = 'gssa' 
		# Load in the model and set up to use the gsl solver.
		modelId = moose.loadModel( './stargazin_synapse.g', 'model', method )
		moose.start( 1000.0 ) # Run the model for 1000 seconds.
		Ca = moose.element( '/model/kinetics/BULK/Ca' )
		Ca.concInit = 2.0e-3 	# Calcium stimulus for turnon
		moose.start( 20.0 ) # Run the model for 100 seconds.
		Ca.concInit = 0.08e-3	# Calcium back to baseline
		moose.start( 1000.0 ) # Let system settle for 1000 sec
		Ca.concInit = 0.2e-3	# 	Calcium turnoff stimulus
		moose.start( 1000.0 ) # Run the model for 1000 seconds for turnoff
		Ca.concInit = 0.08e-3	# Calcium back to baseline
		moose.start( 2000.0 ) # Let system settle for 2000 sec

		displayPlots()
		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
