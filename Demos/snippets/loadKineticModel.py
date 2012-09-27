# This example illustrates loading, running, and saving a kinetic model 
# defined in kkit format. We use the gsl solver here. The model already
# defines a couple of plots and sets the runtime to 100 seconds. The
# script dumps the output into an xplot file called data.plot and the
# saved version into saveReaction.g

import moose
def main():
		# This command loads the file into the path '/model', and tells
		# the system to use the gsl solver.
		modelId = moose.loadModel( '../Genesis_files/reaction.g', 'model', 'gsl' )

		moose.start( 100.0 ) # Run the model for 100 seconds.

		# Iterate through all plots, dump their contents to data.plot.
		for x in moose.wildcardFind( '/model/graphs/conc#/#' ):
				moose.element( x[0] ).xplot( 'data.plot', x[0].name )

		moose.saveModel( modelId, 'saveReaction.g' )
		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
