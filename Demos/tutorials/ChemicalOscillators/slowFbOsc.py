#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2014 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################
# This example illustrates loading, and running a kinetic model 
# for a delayed -ve feedback oscillator, defined in kkit format. 
# The model is one by  Boris N. Kholodenko from 
# Eur J Biochem. (2000) 267(6):1583-8 
# We use the gsl solver here. The model already
# defines some plots and sets the runtime to 4000 seconds.
# The model does not really play nicely with the GSSA solver, since it
# involves some really tiny amounts of the MAPKKK.

import moose
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import numpy
import sys

def main():
        solver = "gsl"
	mfile = '../../Genesis_files/Kholodenko.g'
	runtime = 5000.0
	if ( len( sys.argv ) >= 2 ):
                solver = sys.argv[1]
	modelId = moose.loadModel( mfile, 'model', solver )
        dt = moose.element( '/clock' ).dt

	moose.reinit()
	moose.start( runtime ) 

	# Display all plots.
        img = mpimg.imread( 'Kholodenko_tut.png' )
        fig = plt.figure( figsize=( 12, 10 ) )
        png = fig.add_subplot( 211 )
        imgplot = plt.imshow( img )
        ax = fig.add_subplot( 212 )
	x = moose.wildcardFind( '/model/#graphs/conc#/#' )
        t = numpy.arange( 0, x[0].vector.size, 1 ) * dt
        ax.plot( t, x[0].vector * 100, 'b-', label='Ras-MKKK * 100' )
        ax.plot( t, x[1].vector, 'y-', label='MKKK-P' )
        ax.plot( t, x[2].vector, 'm-', label='MKK-PP' )
        ax.plot( t, x[3].vector, 'r-', label='MAPK-PP' )
        plt.ylabel( 'Conc (mM)' )
        plt.xlabel( 'Time (seconds)' )
        pylab.legend()
        pylab.show()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
