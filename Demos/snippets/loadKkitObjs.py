# This example illustrates loading, running, and saving a kinetic model 
# defined in kkit format. We use the gsl solver here. The model already
# defines a couple of plots and sets the runtime to 100 seconds. The
# script dumps the output into an xplot file called data.plot and the
# saved version into saveReaction.g

import moose
import os
import sys

def main():
		mfile = '../Genesis_files/kkit_objects_example.g'
		runtime = 20.0
		if ( len( sys.argv ) == 3 ):
			mfile = '../Genesis_files/' + sys.argv[1]
			runtime = float( sys.argv[2] )
		modelId = moose.loadModel( mfile, 'model', 'Neutral' )
		"""
		moose.useClock( 4, "/model/kinetics/##[]", "process" )
		moose.setClock( 4, 0.1 )
		moose.useClock( 5, "/model/#graphs/##", "process" )
		moose.setClock( 5, 1 )
		"""
		moose.reinit()

		moose.start( runtime ) 

		# Iterate through all plots, dump their contents to data.plot.
		if ( os.path.exists( 'data.plot' ) ):
			os.remove( 'data.plot' )
		for x in moose.wildcardFind( '/model/#graphs/conc#/#' ):
				moose.element( x ).xplot( 'data.plot', x.name )
				#x.xplot( 'data.plot', x.name )
				#x[0].xplot( 'data.plot', x[0].name )
				#moose.element( x[0] ).xplot( 'data.plot', x[0].name )

		for x in moose.wildcardFind( '/model/kinetics/##[ISA=PoolBase]' ):
			print x.name, x.nInit, x.concInit
			#print x[0].name, x[0].nInit, x[0].concInit
		for x in moose.wildcardFind( '/model/kinetics/##[ISA=ReacBase]' ):
			print x.name, 'num: (', x.numKf, ', ',  x.numKb, '), conc: (', x.Kf, ', ', x.Kb, ')'
			#print x[0].name, 'num: (', x[0].numKf, ', ',  x[0].numKb, '), conc: (', x[0].Kf, ', ', x[0].Kb, ')'
		for x in moose.wildcardFind('/model/kinetics/##[ISA=CplxEnzBase]'):
			print x.name, 'numrates: (', x.k1, ', ',  x.k2, ', ', x.k3, '), conc: (', x.concK1, ', ', x.Km, ')'
			#print x[0].name, 'numrates: (', x[0].k1, ', ',  x[0].k2, ', ', x[0].k3, '), conc: (', x[0].concK1, ', ', x[0].Km, ')'

		for x in moose.wildcardFind('/model/kinetics/##[ISA=MMenz]'):
			print x.name, '(', x.Km, ', ',  x.numKm, ', ', x.kcat, ')'
			#print x[0].name, '(', x[0].Km, ', ',  x[0].numKm, ', ', x[0].kcat, ')'
		print moose.element( '/model/kinetics' ).volume
		# moose.saveModel( modelId, 'saveReaction.g' )
		os.system( "/home/bhalla/code/xplot/xplot data.plot" )
		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
