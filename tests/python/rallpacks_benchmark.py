#!/usr/bin/env python

"""rallpacks_benchmark.py 

    Implements the linear cable with 1000 comparments for Rallpacks
    branchmarking.

Last modified: Sat Jan 18, 2014  05:01PM

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

class RallPacks():

    def __init__(self):
        self.cable = moose.Neutral('/cable')

    def runBenchmarks(self):
        self.runFirstBenchmark()
    
    def runFirstBenchmark(self):
        print("[STEP] First benchmark running ...")
        cable = self.buildCable(1000)

    def buildCable(self, noOfCompartments):
        ''' Build a cable of noOfCompartments comparments '''
        compList = []
        for i in range(noOfCompartments):
            compList.append(moose.Compartment('/cable/comp{}'.format(i)))

        # Connect these compartments linearly.

if __name__ == "__main__":
    rallpacks = RallPacks()
    rallpacks.runBenchmarks()
