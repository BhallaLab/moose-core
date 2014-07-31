#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################

# This example illustrates how to define a kinetic model embedded in
# a NeuroMesh, and undergoing cross-compartment reactions. It is 
# completely self-contained and does not use any external model definition
# files.  Normally one uses standard model formats like
# SBML or kkit to concisely define kinetic and neuronal models.
# This example creates a simple reaction a <==> b <==> c in which 
# a, b, and c are in the dendrite, spine head, and PSD respectively.
# The model is set up to run using the Ksolve for integration. Although
# a diffusion solver is set up, the diff consts here are set to zero.
#

import math
import pylab
import numpy
import matplotlib.pyplot as plt
import moose

def makeCompt( name, parent, dx, dy, dia ):
    RM = 1.0
    RA = 1.0
    CM = 0.01
    EM = -0.065
    pax = 0
    pay = 0
    if ( parent.className == "Compartment" ):
        pax = parent.x
        pay = parent.y
    compt = moose.Compartment( name )
    compt.x0 = pax
    compt.y0 = pay
    compt.z0 = 0
    compt.x = pax + dx
    compt.y = pay + dy
    compt.z = 0
    compt.diameter = dia
    clen = numpy.sqrt( dx * dx + dy * dy )
    compt.length = clen
    compt.Rm = RM / (numpy.pi * dia * clen)
    compt.Ra = RA * 4.0 * numpy.pi * clen / ( dia * dia )
    compt.Cm = CM * numpy.pi * dia * clen
    if ( parent.className == "Compartment" ):
        moose.connect( parent, 'raxial', compt, 'axial' )
    return compt


def makeNeuron( numSeg ):
    segmentLength = 1e-6
    segmentDia = 1e-6
    shaftLength = 1e-6
    shaftDia = 0.2e-6
    headLength = 0.5e-6
    headDia = 0.5e-6
    
    cell = moose.Neutral( '/model/cell' )
    model = moose.element( '/model' )
    prev = makeCompt( '/model/cell/soma', 
            model, 0.0, segmentLength, segmentDia )
    dend = prev
    for i in range( 0, numSeg ):
        name = '/model/cell/dend' + str( i )
        dend = makeCompt( name, dend, 0.0, segmentLength, segmentDia )
        name = '/model/cell/shaft' + str( i )
        shaft = makeCompt( name, dend, shaftLength, 0.0, shaftDia )
        name = '/model/cell/head' + str( i )
        head = makeCompt( name, shaft, headLength, 0.0, headDia )
    return cell

def makeModel():
                numSeg = 5
                diffConst = 0.0
		# create container for model
		model = moose.Neutral( 'model' )
		compt0 = moose.NeuroMesh( '/model/compt0' )
                compt0.separateSpines = 1
                compt0.geometryPolicy = 'cylinder'
		compt1 = moose.SpineMesh( '/model/compt1' )
                moose.connect( compt0, 'spineListOut', compt1, 'spineList', 'OneToOne' )
		compt2 = moose.PsdMesh( '/model/compt2' )
                moose.connect( compt0, 'psdListOut', compt2, 'psdList', 'OneToOne' )

		# create molecules and reactions
		a = moose.Pool( '/model/compt0/a' )
		b = moose.Pool( '/model/compt1/b' )
		c = moose.Pool( '/model/compt2/c' )
		reac0 = moose.Reac( '/model/compt0/reac0' )
		reac1 = moose.Reac( '/model/compt1/reac1' )

		# connect them up for reactions
		moose.connect( reac0, 'sub', a, 'reac' )
		moose.connect( reac0, 'prd', b, 'reac' )
		moose.connect( reac1, 'sub', b, 'reac' )
		moose.connect( reac1, 'prd', c, 'reac' )

		# Assign parameters
		a.diffConst = diffConst
		b.diffConst = diffConst
		c.diffConst = diffConst
		a.concInit = 1
		b.concInit = 12.1
		c.concInit = 1
		reac0.Kf = 1
		reac0.Kb = 1
		reac1.Kf = 1
		reac1.Kb = 1
                print reac0.numKf, reac0.numKb
                print reac1.numKf, reac1.numKb
                print a.volume, b.volume, c.volume

                # Create a 'neuron' with a dozen spiny compartments.
                elec = makeNeuron( numSeg )
                # assign geometry to mesh
                compt0.diffLength = 10e-6
                compt0.cell = elec
                print reac0.numKf, reac0.numKb
                print reac1.numKf, reac1.numKb
                print a.vec.volume, b.vec.volume, c.vec.volume

                # Build the solvers. No need for diffusion in this version.
                ksolve0 = moose.Ksolve( '/model/compt0/ksolve' )
                ksolve1 = moose.Ksolve( '/model/compt1/ksolve' )
                ksolve2 = moose.Ksolve( '/model/compt2/ksolve' )
                #dsolve0 = moose.Dsolve( '/model/compt0/dsolve' )
                #dsolve1 = moose.Dsolve( '/model/compt1/dsolve' )
                #dsolve2 = moose.Dsolve( '/model/compt2/dsolve' )
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
                #stoich0.dsolve = dsolve0
                #stoich1.dsolve = dsolve1
                #stoich2.dsolve = dsolve2
                stoich0.path = '/model/compt0/#'
                stoich1.path = '/model/compt1/#'
                stoich2.path = '/model/compt2/#'
                assert( stoich0.numVarPools == 1 )
                assert( stoich0.numProxyPools == 1 )
                assert( stoich0.numRates == 1 )
                assert( stoich1.numVarPools == 1 )
                assert( stoich1.numProxyPools == 1 )
                assert( stoich1.numRates == 1 )
                assert( stoich2.numVarPools == 1 )
                assert( stoich2.numProxyPools == 0 )
                assert( stoich2.numRates == 0 )
                #dsolve0.buildNeuroMeshJunctions( dsolve1, dsolve2 )
                stoich0.buildXreacs( stoich1 )
                #stoich1.buildXreacs( stoich0 )
                stoich1.buildXreacs( stoich2 )
                print a.vec.volume, b.vec.volume, c.vec.volume
		a.vec.concInit = range( numSeg + 1, 0, -1 )
		b.vec.concInit = [5.0 * ( 1 + x ) for x in range( numSeg )]
		c.vec.concInit = range( 1, numSeg + 1 )
		#a.vec.concInit = [2] * (numSeg + 1)
		#b.vec.concInit = [10] * numSeg
		#c.vec.concInit = [1] * numSeg
                print a.vec.concInit, b.vec.concInit, c.vec.concInit

		# Create the output tables
		graphs = moose.Neutral( '/model/graphs' )
		outputA = moose.Table ( '/model/graphs/concA' )
		outputB = moose.Table ( '/model/graphs/concB' )
		outputC = moose.Table ( '/model/graphs/concC' )

		# connect up the tables
                a1 = moose.element( '/model/compt0/a[2]' )
                b1 = moose.element( '/model/compt1/b[1]' )
                c1 = moose.element( '/model/compt2/c[1]' )
		moose.connect( outputA, 'requestOut', a1, 'getConc' );
		moose.connect( outputB, 'requestOut', b1, 'getConc' );
		moose.connect( outputC, 'requestOut', c1, 'getConc' );


def main():
                simdt = 0.01
                plotdt = 0.01

		makeModel()

		# Schedule the whole lot
		moose.setClock( 4, simdt ) # for the computational objects
		moose.setClock( 5, simdt ) # for the computational objects
		moose.setClock( 8, plotdt ) # for the plots
		#moose.useClock( 4, '/model/compt#/dsolve', 'process' )
		moose.useClock( 4, '/model/compt#/ksolve', 'init' )
		moose.useClock( 5, '/model/compt#/ksolve', 'process' )
		moose.useClock( 8, '/model/graphs/#', 'process' )

                a = moose.element( '/model/compt0/a' )
                b = moose.element( '/model/compt1/b' )
                c = moose.element( '/model/compt2/c' )
		reac0 = moose.element( '/model/compt0/reac0' )
		reac1 = moose.element( '/model/compt1/reac1' )
                print "reac0 kf,kb: ", reac0.vec.numKf, reac0.vec.numKb
                print "reac1 kf,kb: ", reac1.vec.numKf, reac1.vec.numKb
                print "reac0 Kf,Kb: ", reac0.vec.Kf, reac0.vec.Kb
                print "reac1 Kf,Kb: ", reac1.vec.Kf, reac1.vec.Kb

		moose.reinit()
                print "a: ", a.vec.n
                print "b: ", b.vec.n
                print "c: ", c.vec.n
		#moose.start( 10.0 ) # Run the model for 100 seconds.
                display()
                print "a: ", a.vec.conc
                print "b: ", b.vec.conc
                print "c: ", c.vec.conc

		# Iterate through all plots, dump their contents to data.plot.
                '''
		for x in moose.wildcardFind( '/model/graphs/conc#' ):
				#x.xplot( 'scriptKineticModel.plot', x.name )
				t = numpy.arange( 0, x.vector.size, 1 ) # sec
				pylab.plot( t, x.vector, label=x.name )
		pylab.legend()
		pylab.show()
                '''
		quit()

def display():
    dt = 0.01
    runtime = 1
    a = moose.element( '/model/compt0/a' )
    b = moose.element( '/model/compt1/b' )
    c = moose.element( '/model/compt2/c' )
    plt.ion()
    fig = plt.figure( figsize=(12,10))
    timeseries = fig.add_subplot( 211 )
    spatial = fig.add_subplot( 212)
    spatial.set_ylim(0, 15)
    pos = numpy.arange( 0, a.vec.conc.size, 1 )
    line1, = spatial.plot( pos, a.vec.conc, 'b-', label='a' )
    line2, = spatial.plot( pos[1:], b.vec.conc, 'g-', label='b' )
    line3, = spatial.plot( pos[1:], c.vec.conc, 'r-', label='c' )
    timeLabel = plt.text( 3, 12, 'time = 0' )
    plt.legend()
    fig.canvas.draw()

    for t in numpy.arange( dt, runtime, dt ):
        line1.set_ydata( a.vec.conc )
        line2.set_ydata( b.vec.conc )
        line3.set_ydata( c.vec.conc )
        #print b.vec.volume
        #print a.vec.n, b.vec.n, c.vec.n
        timeLabel.set_text( "time = %f" % t )
        fig.canvas.draw()
        #raw_input()
        moose.start( dt )

    timeseries.set_ylim( 0, 12 )
    for x in moose.wildcardFind( '/model/graphs/conc#' ):
        t = numpy.arange( 0, x.vector.size *dt , dt ) # sec
        line4, = timeseries.plot( t, x.vector, label=x.name )
    plt.legend()
    fig.canvas.draw()

    print( "Hit 'enter' to exit" )
    raw_input()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
