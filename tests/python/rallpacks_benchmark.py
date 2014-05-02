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

sys.path.append('../../python')
import moose
import moose.utils
import pylab

simTime = 100
simDt = 1e-2
plotDt = 0.5e-2
cableSize = 10

def connectTwoComp(comp1, comp2):
    print("[CONNECT] Connecting {} and {}".format(comp1.path, comp2.path))
    comp1.connect('raxial', comp2, 'axial')

class RallPacks():

    def __init__(self):
        moose.Neutral('/cable')
        self.cable = []

    def runBenchmarks(self):
        self.runFirstBenchmark()
        #self.runSecondBenchMark()
        #self.runThirdBenchMark()
    
    def runFirstBenchmark(self):
        print("[STEP] Setting up first benchmark... ", endby='')
        self.setupFirstBenchmarks()
        print("done")

    def setupFirstBenchmarks(self):
        self.buildCable(cableSize)
        outputTable, inputTable = self.setupDUT()


    def buildCable(self, noOfCompartments):
        ''' Build a cable of noOfCompartments comparments '''
        for i in range(noOfCompartments):
            comp = moose.Compartment("/cable/comp{}".format(i))
            comp.Rm = 7.6e6
            comp.Cm = 7e-9
            self.cable.append(comp)
        # Connect these compartments linearly.
        for c in range(1, len(self.cable)):
            connectTwoComp(self.cable[c-1], self.cable[c])

    def setupDUT(self):
        ''' Attach a pulse generator at the first compartment and record from
        the last'''
        pulse = moose.PulseGen('/cable/pulsegen')
        pulse.delay[0] = 2
        pulse.width[0] = 10
        pulse.level[0] = 3e-6
        moose.connect(pulse, 'output', self.cable[0], 'injectMsg')

        # Do recording here
        moose.Neutral('/data')
        recordingAtStim = moose.Table('/data/stim')
        moose.connect(recordingAtStim, 'requestOut', pulse, 'getOutputValue')
        vmTab = moose.Table('/data/cable_vm')
        moose.connect(vmTab, 'requestOut', self.cable[1], 'getVm')
        return vmTab, recordingAtStim

    def setupClocks(self):
        """ Setup clocks for simulation 
        """
        moose.setClock(0, 1e-3)
        moose.setClock(1, 0.5e-3)
        moose.useClock(0, '/cable/##', 'process')

        # Initializing is very important
        moose.useClock(0, '/cable/##', 'init')
        moose.useClock(1, '/data/#[TYPE=Table]', 'process')
        moose.reinit()

def main():
    rallpacks = RallPacks()
    rallpacks.runBenchmarks()

if __name__ == "__main__":
    main()

