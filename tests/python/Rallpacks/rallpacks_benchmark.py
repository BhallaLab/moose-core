#!/usr/bin/env python

"""rallpacks_benchmark.py 

    Implements the linear cable with 1000 comparments for Rallpacks
    branchmarking.

Last modified: Fri May 02, 2014  11:23AM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import sys
import os
import time

sys.path.append('../../python')
import moose
import moose.utils
import pylab

startTime = time.time()
simTime = 100
simDt = 1e-2
plotDt = 0.5e-2

# Record from compartments at stepSize away
cableSize = 100
stepSize = int(cableSize / 10)

def connectTwoComp(comp1, comp2):
    """ Connecting two compartments """
    print("[CONNECT] Connecting {} and {}".format(comp1.path, comp2.path))
    comp1.connect('raxial', comp2, 'axial')

class RallPacks():
    ''' Run Rallpacks. '''

    def __init__(self):
        moose.Neutral('/cable')
        moose.Neutral('/data')
        self.cable = []

    def runBenchmarks(self):
        ''' Run all benchmarks 
        '''
        self.runFirstBenchmark()
        #self.runSecondBenchMark()
        #self.runThirdBenchMark()
    
    def runFirstBenchmark(self):
        print("[STEP] Setting up first benchmark... ")
        inputTable = self.setupFirstBenchmarks()
        self.setupClocks()
        moose.start(simTime)
        print("[RALLPACKS] Cable with {} compartments took {} sec.".format(
            cableSize
            , time.time() - startTime)
            )
        moose.utils.plotTable(inputTable, file='input.eps')
        referenceVec = self.tables[0].vector
        for t in self.tables[1:]:
            pylab.plot( t.vector - referenceVec )
        pylab.savefig('rallpacks1.eps')

    def setupFirstBenchmarks(self):
        ''' Setup cable with cableSize compartments.
        Return input and output tables after simulation.
        '''
        self.buildCable(cableSize)

        # Pulse for input
        pulse = moose.PulseGen('/cable/pulsegen')
        pulse.delay[0] = 10
        pulse.width[0] = 20
        pulse.level[0] = 3e-5
        moose.connect(pulse, 'output', self.cable[0], 'injectMsg')

        # Read the input from stimulus
        stimTable = moose.Table('/data/stim')
        moose.connect(stimTable, 'requestOut', pulse, 'getOutputValue')

        # Do recording at each compartment.
        self.tables = []
        for i in range(1, len(self.cable), stepSize):
            table = moose.Table('/data/table{}'.format(i))
            moose.connect(table, 'requestOut', self.cable[i], 'getVm')
            self.tables.append(table)

        return stimTable


    def buildCable(self, noOfCompartments):
        ''' Build a cable of noOfCompartments comparments '''
        for i in range(noOfCompartments):
            comp = moose.Compartment("/cable/comp{}".format(i))
            #comp.Rm = 1e6
            #comp.Cm = 7e-9
            self.cable.append(comp)
        # Connect these compartments linearly.
        for c in range(1, len(self.cable)):
            connectTwoComp(self.cable[c-1], self.cable[c])

    def setupClocks(self):
        """ Setup clocks for simulation 
        """
        moose.setClock(0, 1e-3)
        moose.setClock(1, 0.5e-3)
        moose.useClock(0, '/cable/##', 'process')

        # Initializing is very important
        moose.useClock(0, '/cable/##', 'init')
        moose.useClock(1, '/data/##', 'process')
        moose.reinit()

def main():
    rallpacks = RallPacks()
    rallpacks.runBenchmarks()

if __name__ == "__main__":
    main()

