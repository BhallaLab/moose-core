#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2014 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################
# This example illustrates loading, and running a kinetic model 
# for a relaxation oscillator, defined in kkit format.
# It uses the deterministic gsl solver for starters, you can specify 
# another the stochastic Gillespie solver the command line 
#     python filename gssa
# We use the gsl solver here. The model already
# defines some plots and sets the runtime to 4000 seconds.

import moose
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import numpy
import sys

def main():
        solver = "gsl"  # Pick any of gsl, gssa, ee..
        #solver = "gssa"  # Pick any of gsl, gssa, ee..
	mfile = '../../Genesis_files/OSC_Cspace.g'
	runtime = 4000.0
	if ( len( sys.argv ) >= 2 ):
                solver = sys.argv[1]
	modelId = moose.loadModel( mfile, 'model', solver )
        # Increase volume so that the stochastic solver gssa 
        # gives an interesting output
        compt = moose.element( '/model/kinetics' )
        compt.volume = 1e-19 
        dt = moose.element( '/clock' ).tickDt[18] # 18 is the plot clock.

	moose.reinit()
	moose.start( runtime ) 

	# Display all plots.
        img = mpimg.imread( 'relaxOsc_tut.png' )
        fig = plt.figure( figsize=(12, 10 ) )
        png = fig.add_subplot( 211 )
        imgplot = plt.imshow( img )
        ax = fig.add_subplot( 212 )
	x = moose.wildcardFind( '/model/#graphs/conc#/#' )
        t = numpy.arange( 0, x[0].vector.size, 1 ) * dt
        ax.plot( t, x[0].vector, 'b-', label=x[0].name )
        ax.plot( t, x[1].vector, 'c-', label=x[1].name )
        ax.plot( t, x[2].vector, 'r-', label=x[2].name )
        ax.plot( t, x[3].vector, 'm-', label=x[3].name )
        plt.ylabel( 'Conc (mM)' )
        plt.xlabel( 'Time (seconds)' )
        pylab.legend()
        pylab.show()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
