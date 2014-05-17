#!/usr/bin/env python

"""cable.py: A passive cable of n compartments.

Last modified: Thu May 15, 2014  05:49PM

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

    def __init__(self, length, compartmentSize):
        ''' Initialize the cable '''
        self.save_dir = 'figures'
        if not os.path.isdir( self.save_dir ):
            os.makedirs( self.save_dir )

        self.length = length
        self.compartmentSize = compartmentSize
        self.nseg = int(self.length / self.compartmentSize)
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
        self.makeCable()

    def makeCable( self, **compartment_options ):
        ''' Make a cable out of n compartments '''
        for i in range( self.nseg ):
            c = comp.MooseCompartment( **compartment_options )
            c.createCompartment( 
                    length = self.compartmentSize
                    , diameter = 1e-6
                    , path = '%s/comp_%s' % ( self.cablePath , i) 
                    )
            self.cable.append(c)
        self.connect( )
        utils.dump( "STEP"
                , "Passive cable is connected and ready for simulation." 
                )

    def connect( self ):
        ''' Connect the cable '''
        utils.dump('STEP', 'Connecting cable ...')
        for i, c1 in enumerate( self.cable[:-1] ):
            c2 = self.cable[i+1].mc_
            c1.mc_.connect('axial', c2, 'raxial')

    def setupDUT( self ):
        ''' Setup cable for recording '''

        # Create a pulse input
        moose.Neutral( self.tablePath )
        stim = moose.PulseGen( '{}/input'.format( self.tablePath) )
        stim.level[0] = 0.1e-9
        stim.width[0] = 0.25
        stim.delay[0] = 0.0

        # Inject the current from stim to first compartment.
        moose.connect( stim, 'output', self.cable[0].mc_, 'injectMsg' )
        
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
        moose.setClock( 1, self.simDt )
        moose.setClock( 2, self.simDt )

        ## Use clocksc
        moose.useClock( 0, '/cable/##'.format(self.cablePath), 'process' )
        moose.useClock( 1, '/cable/##'.format(self.cablePath), 'init' )
        moose.useClock( 2, '{}/##'.format(self.tablePath), 'process' )

        utils.dump("STEP"
                , [ "Simulating cable for {} sec".format(simTime)
                    , " simDt: %s, plotDt: %s" % ( self.simDt, self.plotDt )
                    ]
                )
        moose.reinit( )
        utils.verify( )
        moose.start( simTime )

def main( ):
    cableLength = 1.0e-3
    compNons = int(sys.argv[1])
    outputFile = None
    try:
        outputFile = sys.argv[2]
    except: pass
    compartmentSize = cableLength / compNons
    cable = Cable( cableLength, compartmentSize)
    first = 0
    last = cable.nseg - 1
    middle = ( first + last + 1 ) / 2
    table1 = utils.recordTarget('/data/table1', cable.cable[first].mc_, 'vm' )
    table2 = utils.recordTarget('/data/table2', cable.cable[middle].mc_, 'vm' )
    table3 = utils.recordTarget('/data/table3', cable.cable[last].mc_, 'vm' )
    sim_dt = 1e-4
    cable.simulate( simTime = 0.25, simDt = sim_dt )
    utils.plotTables([table1, table2, table3]
            , xscale = sim_dt
            , file = outputFile
            )
    import moose.backend.spice as spice
    spice.toSpiceNetlist( output = 'cable.spice' )

if __name__ == '__main__':
    main()

