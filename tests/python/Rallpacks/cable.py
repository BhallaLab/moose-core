#!/usr/bin/env python

"""cable.py: A passive cable of n compartments.

Last modified: Wed May 07, 2014  11:38PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"


import compartment as comp
import sys
sys.path.append('../../python/')
import moose
import moose.utils as utils
import os
import pylab
import numpy as np

class Cable( ):
    ''' Class representing a cable '''

    def __init__( self, length = 1e-3, compartmentSize = 1e-6 ):
        ''' Initialize the cable '''
        self.length = length
        self.compartmentSize = compartmentSize
        utils.dump("INFO"
                , [ "Creating a cable."
                    , "length: %s" % self.length
                    , "size: %s" % self.compartmentSize 
                    ]
                )
        self.size = int(self.length / self.compartmentSize)
        self.cablePath = '/cable'
        self.tablePath = '/data'
        moose.Neutral( self.cablePath )
        moose.Neutral( self.tablePath )

        # Store all moose-compartments in this list.
        self.cable = []
        # Keep all simulation data in this list of tables.
        self.tables = []
        # keep all stimulus in this list of tables.
        self.stimTables = []

    def makeCable( self ):
        ''' Make a cable out of n compartments '''
        for i in range( self.size ):
            c = comp.MooseCompartment( )
            c.createCompartment( path = '%s/comp_%s' % ( self.cablePath , i) )
            self.cable.append( c )
        self.connect( )
        utils.dump( "INFO"
                , "Passive cable is connected and ready for simulation." 
                )

    def connect( self ):
        ''' Connect the cable '''
        utils.dump('STEP', 'Connecting cable ...')
        for i, c1 in enumerate( self.cable[:-1] ):
            c2 = self.cable[i+1].mooseCompartment
            #c1.mooseCompartment.connect('raxial', c2.mooseCompartment, 'axial')
            c1.mooseCompartment.connect('raxial', c2, 'axial')

    def recordAt( self, index ):
        ''' Parameter index is python list-like index. Index -1 is the last
        elements in the list 
        '''
        utils.dump( "RECORD", "Setting up a probe at compartment no %s " % index )
        if index < 0:
            index = len( self.cable ) + index
        if( index >= len( self.cable ) ):
            raise UserWarning( "There is no compartment at index %s" % index )
        #c = self.cable[ index ].mooseCompartment 
        c = self.cable[index].mooseCompartment
        t = moose.Table( '{}/output_at_{}'.format( self.tablePath, index ))
        moose.connect( t, 'requestOut', c, 'getVm' )
        self.tables.append( t )

    def setupDUT( self ):
        ''' Setup cable for recording '''

        # Create a pulse input
        moose.Neutral( self.tablePath )
        stim = moose.PulseGen( '{}/input'.format( self.tablePath) )
        stim.level[0] = 1e-9
        stim.width[0] = 0.2
        stim.delay[0] = 0.1
        stim.delay[1] = 0.8

        # Inject the current from stim to first compartment.
        moose.connect( stim, 'output', self.cable[0].mooseCompartment, 'injectMsg' )
        
        # Fill the data from stim into table.
        inputTable = moose.Table( '{}/inputTable'.format( self.tablePath ) )
        self.stimTables.append( inputTable )
        moose.connect( inputTable, 'requestOut', stim, 'getOutputValue' )

    def simulate( self, simTime, simDt = 1e-3, plotDt = None ):
        '''Simulate the cable '''

        if plotDt is None:
            plotDt = simDt / 2
        self.simDt = simDt
        self.plotDt = plotDt
        self.setupDUT( )
 
        # Setup clocks 
        utils.dump("STEP", "Setting up the clocks ... ")
        moose.setClock( 0, self.simDt )
        moose.setClock( 1, self.plotDt )

        # Use clocks
        moose.useClock( 0, '/##'.format(self.cablePath), 'process' )
        moose.useClock( 0, '/##'.format(self.cablePath), 'init' )
        #moose.useClock( 0, '{}/##'.format(self.tablePath), 'process' )

        utils.dump("STEP"
                , [ "Simulating cable for {} sec".format(simTime)
                    , " simDt: %s, plotDt: %s" % ( self.simDt, self.plotDt )
                    ]
                )
        moose.reinit( )
        moose.verify( )
        moose.start( simTime )

    def plotTables( self, ascii = False ):
        ''' Plot all tables: stimulus and recording. '''
        if not ascii:
            pylab.figure( )
        [ self.plotTable(t, self.simDt, ascii=ascii) for t in self.stimTables ]
        if not ascii:
            pylab.figure( )
        [ self.plotTable(t, self.simDt, ascii=ascii) for t in self.tables ]
        if not ascii:
            pylab.show( )

    def plotTable( self, table, dt, standalone = False, ascii = False):
        ''' Plot a single table '''
        if standalone and not ascii:
            pylab.figure( )
        yvec = table.vector
        # Multiply index with simDt to get the time at which recording was
        # made.
        xvec = [ x * dt for x in range(len(table.vector)) ]
        if not ascii:
            pylab.plot( xvec, yvec )
            pylab.xlabel( 'Time (sec)' )
            pylab.legend( '{}'.format(table.path) )
        else:
            utils.plotAscii( yvec, xvec )

    def checkResults( self ):
        ''' Check if tables make any sense '''
        outputTables = self.tables[:]
        t1 = outputTables.pop(0)
        while len(outputTables) > 0:
            t2 = outputTables.pop(0)
            while t2.path == t1.path:
                printMsg("WARN", "Two tables with same path. Ignoring ...")
                t2 = outputTables.pop(0)
            y1 = t1.vector
            y2 = t2.vector 
            if np.array_equal(y1, y2):
                utils.dump("ERROR"
                        , "Something fishy."
                        , "Same data found at two differenct compartments."
                        )
            elif np.greater( y1, y2 ).all():
                utils.dump( "ERROR"
                        , "Something fishy"
                        , "First table must not be greater than the second one"
                        )
            else: 
                utils.dump("INFO", "Tables looks OK. Go for plotting... ")
            t2 = t1

    def solveAnalytically( self ):
        ''' Solve the cable analytically at a position x for all time t in
        simTime '''
        utils.dump("WARNING", "This solves it assuming that input is a step "
                "function of 1 nA of current."
                )
def main( ):
    cable = Cable( length = 1e-3, compartmentSize = 1e-6 )
    cable.makeCable( )
    cable.recordAt( index = 0 )
    cable.recordAt( index = -1 )
    cable.simulate( simTime = 10, simDt = 1e-3 )
    cable.checkResults( )
    cable.plotTables( ascii = False )
    #cable.solveAnalytically( )

if __name__ == '__main__':
    main()

