#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################

# This example illustrates how to define a kinetic model using the
# scripting interface. Normally one uses standard model formats like
# SBML or kkit to concisely define kinetic models, but in some cases one
# would like to modify the model through the script.
# This example creates a bistable model having two enzymes and a reaction.
# One of the enzymes is autocatalytic.
# The model is set up to run using default Exponential Euler integration.
# The snippet scriptKineticSolver.py uses the much better GSL 
# Runge-Kutta-Fehlberg integration scheme on this same model.

import math
import pylab
import numpy
import moose

def makeModel():
		# create container for model
		model = moose.Neutral( 'model' )
		compt0 = moose.CubeMesh( '/model/compt0' )
		compt0.volume = 1e-15
		compt1 = moose.CubeMesh( '/model/compt1' )
		compt1.volume = 1e-16
		compt2 = moose.CubeMesh( '/model/compt2' )
		compt2.volume = 1e-17

                # Position containers so that they abut each other, with
                # compt1 in the middle.
                print compt1.dx, compt1.dy, compt1.dz
                side = compt1.dy
                moose.showfields( compt0 )
                compt0.y1 += side
                compt0.y0 += side
                compt2.x1 += side
                compt2.x0 += side
                moose.showfields( compt0 )
                print compt0.volume, compt1.volume, compt2.volume

		# create molecules and reactions
		a = moose.Pool( '/model/compt0/a' )
		b = moose.Pool( '/model/compt1/b' )
		c = moose.Pool( '/model/compt2/c' )
		reac0 = moose.Reac( '/model/compt1/reac0' )
		reac1 = moose.Reac( '/model/compt1/reac1' )

		# connect them up for reactions
		moose.connect( reac0, 'sub', a, 'reac' )
		moose.connect( reac0, 'prd', b, 'reac' )
		moose.connect( reac1, 'sub', b, 'reac' )
		moose.connect( reac1, 'prd', c, 'reac' )

		# Assign parameters
		a.concInit = 1
		b.concInit = 12.1
		c.concInit = 1
		reac0.Kf = 0.1
		reac0.Kb = 0.1
		reac1.Kf = 0.1
		reac1.Kb = 0.1
                print reac0.numKf, reac0.numKb
                print reac1.numKf, reac1.numKb
                print a.volume, b.volume, c.volume

		# Create the output tables
		graphs = moose.Neutral( '/model/graphs' )
		outputA = moose.Table ( '/model/graphs/concA' )
		outputB = moose.Table ( '/model/graphs/concB' )
		outputC = moose.Table ( '/model/graphs/concC' )

		# connect up the tables
		moose.connect( outputA, 'requestOut', a, 'getConc' );
		moose.connect( outputB, 'requestOut', b, 'getConc' );
		moose.connect( outputC, 'requestOut', c, 'getConc' );

                # Build the solvers. No need for diffusion in this version.
                ksolve0 = moose.Ksolve( '/model/compt0/ksolve' )
                ksolve1 = moose.Ksolve( '/model/compt1/ksolve' )
                ksolve2 = moose.Ksolve( '/model/compt2/ksolve' )
                stoich0 = moose.Stoich( '/model/compt0/stoich' )
                stoich1 = moose.Stoich( '/model/compt1/stoich' )
                stoich2 = moose.Stoich( '/model/compt2/stoich' )

                # Configure solvers
                stoich0.compartment = compt0
                stoich1.compartment = compt1
                stoich2.compartment = compt2
                stoich0.ksolve = ksolve0
                stoich1.ksolve = ksolve1
                stoich2.ksolve = ksolve2
                stoich0.path = '/model/compt0/#'
                stoich1.path = '/model/compt1/#'
                stoich2.path = '/model/compt2/#'
                assert( stoich0.numVarPools == 1 )
                assert( stoich0.numProxyPools == 0 )
                assert( stoich0.numRates == 0 )
                assert( stoich1.numVarPools == 1 )
                assert( stoich1.numProxyPools == 2 )
                assert( stoich1.numRates == 2 )
                assert( stoich2.numVarPools == 1 )
                assert( stoich2.numProxyPools == 0 )
                assert( stoich2.numRates == 0 )
                stoich0.buildXreacs( stoich1 )
                #stoich1.buildXreacs( stoich0 )
                stoich1.buildXreacs( stoich2 )



def main():
                simdt = 0.1
                plotdt = 0.1

		makeModel()

		# Schedule the whole lot
		moose.setClock( 4, simdt ) # for the computational objects
		moose.setClock( 5, simdt ) # for the computational objects
		moose.setClock( 8, plotdt ) # for the plots
		moose.useClock( 4, '/model/compt#/ksolve', 'init' )
		moose.useClock( 5, '/model/compt#/ksolve', 'process' )
		moose.useClock( 8, '/model/graphs/#', 'process' )

		moose.reinit()
		moose.start( 100.0 ) # Run the model for 100 seconds.
		for x in moose.wildcardFind( '/model/compt#/#[ISA=PoolBase]' ):
                    print x.name, x.conc

		# Iterate through all plots, dump their contents to data.plot.
		for x in moose.wildcardFind( '/model/graphs/conc#' ):
				#x.xplot( 'scriptKineticModel.plot', x.name )
				t = numpy.arange( 0, x.vector.size, 1 ) # sec
				pylab.plot( t, x.vector, label=x.name )
		pylab.legend()
		pylab.show()


		quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
