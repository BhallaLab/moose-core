# This example illustrates the transition from deterministic to 
# stochastic in different methods and different simulation volumes.
# See [[./loadKineticModel.py]] for the basic loading process.
# This example first loads and runs the model using the gsl solver. Then it
# switches to the Gillespie gssa method and runs it again at different
# volumes.
# The script dumps the output into an xplot file called kineticSolvers.plot.

import moose
def main():
		# Load in the model and set up to use the gsl solver.
		modelId = moose.loadModel( '../Genesis_files/reaction.g', 'model', 'gsl' )

		moose.start( 100.0 ) # Run the model for 100 seconds.

		# Iterate through all plots, dump their contents to data.plot.
		for x in moose.wildcardFind( '/model/graphs/conc#/#' ):
			moose.element( x[0] ).xplot( 'kineticSolvers.plot', x[0].name + "gsl" )

		# switch methods
		modelId[0].method = 'gssa'
		for vol in ( 1e-20, 1e-19, 1e-18, 1e-17 ):
			moose.element( '/model/kinetics' ).volume = vol
			moose.reinit() # Start over from time zero
			moose.start( 100.0 ) # Run the model for 100 seconds.

			# Dump plots again, with a slightly modified name.
			for x in moose.wildcardFind( '/model/graphs/conc#/#' ):
				name = 'gssa_' + x[0].name + '_' + str( vol )
				moose.element( x[0] ).xplot( 'kineticSolvers.plot', name )

		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
