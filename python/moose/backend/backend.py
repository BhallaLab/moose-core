#!/usr/bin/env python

"""backend.py: Convert moose data-structure to some other backend.

Last modified: Mon May 12, 2014  11:21PM

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
thisDir = os.path.dirname( __file__ )
sys.path.append( os.path.join( thisDir, '..') )

import _moose 
import print_utils as debug
from collections import defaultdict

# Plot in terminal uses gnuplot. So use it as a backend.
import plot_utils 
plotInTerminal = plot_utils.plotInTerminal 
plotAscii = plot_utils.plotAscii


class Backend(object):
    """ Base class for all backend """

    def __init__(self, *args):
        super(Backend, self).__init__()
        self.args = args
        self.compartments = []
        self.pulseGens = []
        self.tables = []
        # A set of tuple of sourceCompartment.path and targetCompartment.path
        self.connections = set()
        self.clock = _moose.wildcardFind('/clock')[0]

    def getComparments(self, **kwargs):
        '''Get all compartments in moose '''
        comps = _moose.wildcardFind('/##[TYPE=Compartment]')
        self.compartments = comps

    def getPulseGens(self, **kwargs):
        """ Get all the pulse generators """
        self.pulseGens = _moose.wildcardFind('/##[TYPE=PulseGen]')

    def getTables(self, **kwargs):
        """ Get all table we are recording from"""
        self.tables = _moose.wildcardFind('/##[TYPE=Table]')

    def populateStoreHouse(self, **kwargs):
        """ Populate all data-structures related with Compartments, Tables, and
        pulse generators.
        """
        debug.dump("INFO", "Populating data-structures to write spice netlist")
        self.getComparments(**kwargs)
        self.getTables(**kwargs)
        self.getPulseGens(**kwargs)
