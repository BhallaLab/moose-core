#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2014 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################
# This example illustrates loading, running, and saving a kinetic model 
# defined in kkit format. It uses a default kkit model but you can
# specify another using the command line 
#     python filename runtime solver
# We use the gsl solver here. The model already
# defines a couple of plots and sets the runtime to 100 seconds. The
# script dumps the output into an xplot file called data.plot and the
# saved version into saveReaction.g
# Since it is dealing with kkit models which may span compartments, the
# 'solver' argument is prefixed with 'old_'. This tells the readKkit
# function to put the entire model tree on the 'kinetics' compartment
# which confuses the volume scaling but lets the solver handle the
# entire model.

import moose
import pylab
import numpy
import sys

def main():
        solver = "old_gsl"  # Pick any of gsl, gssa, ee..
	mfile = '../Genesis_files/kkit_objects_example.g'
	runtime = 20.0
	if ( len( sys.argv ) >= 3 ):
		mfile = '../Genesis_files/' + sys.argv[1]
		runtime = float( sys.argv[2] )
	if ( len( sys.argv ) == 4 ):
                solver = sys.argv[3]
	modelId = moose.loadModel( mfile, 'model', solver )
        # Increase volume so that the stochastic solver gssa 
        # gives an interesting output
        #compt = moose.element( '/model/kinetics' )
        #compt.volume = 1e-19 

	moose.reinit()
	moose.start( runtime ) 

        # Report parameters
        '''
	for x in moose.wildcardFind( '/model/kinetics/##[ISA=PoolBase]' ):
		print x.name, x.nInit, x.concInit
	for x in moose.wildcardFind( '/model/kinetics/##[ISA=ReacBase]' ):
		print x.name, 'num: (', x.numKf, ', ',  x.numKb, '), conc: (', x.Kf, ', ', x.Kb, ')'
	for x in moose.wildcardFind('/model/kinetics/##[ISA=EnzBase]'):
		print x.name, '(', x.Km, ', ',  x.numKm, ', ', x.kcat, ')'
                '''

	# Display all plots.
	for x in moose.wildcardFind( '/model/#graphs/conc#/#' ):
            t = numpy.arange( 0, x.vector.size, 1 ) * x.dt
            pylab.plot( t, x.vector, label=x.name )
        pylab.legend()
        pylab.show()

	quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
