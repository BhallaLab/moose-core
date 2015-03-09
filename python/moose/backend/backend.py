# This file is part of MOOSE simulator: http://moose.ncbs.res.in.

# MOOSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MOOSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with MOOSE.  If not, see <http://www.gnu.org/licenses/>.


"""backend.py: 

Last modified: Wed Feb 11, 2015  06:01PM
"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


import sys
import os
from .. import _moose 
from .. import print_utils 
from collections import defaultdict

class Backend(object):
    """ Base class for all backend """

    def __init__(self, *args):
        super(Backend, self).__init__()
        self.args = args
        self.compartments = []
        self.chemEntities = []
        self.pulseGens = []
        self.tables = []
        self.synchans = []
        self.msgs = { 'SingleMsg' : [], 'OneToAllMsg' : [] }
        # A set of tuple of sourceCompartment.path and targetCompartment.path
        self.connections = set()
        self.clock = _moose.wildcardFind('/clock')[0]
        self.clocks = []
        self.filled = False
        self.root = '/'

    def filterPaths(self, mooseObjs, ignorePat):
        """Filter paths """
        def ignore(x):
            if ignorePat.search(x.path):
                return False
            return True
        if ignorePat:
            mooseObjs = filter(ignore, mooseObjs)
        return mooseObjs

    def getComparments(self, **kwargs):
        '''Get all compartments in moose '''
        self.compartments = _moose.wildcardFind('%s/##[TYPE=Compartment]'%self.root)
        zombiComps = _moose.wildcardFind('%s/##[TYPE=ZombieCompartment]'%self.root)
        if zombiComps:
            self.compartments += zombiComps
        return self.compartments

    def getPulseGens(self, **kwargs):
        """ Get all the pulse generators """
        self.pulseGens = _moose.wildcardFind('%s/##[TYPE=PulseGen]'%self.root)
        return self.pulseGens

    def getTables(self, **kwargs):
        """ Get all table we are recording from"""
        self.tables = _moose.wildcardFind('%s/##[TYPE=Table]'%self.root)
        return self.tables

    def getSynChans(self, **kwargs):
        """Get all the SynChans """
        self.synchans = _moose.wildcardFind('%s/##[TYPE=SynChan]'%self.root)
        return self.synchans

    def getClocks(self, **kwargs):
        """Get all clocks"""
        self.clocks = _moose.wildcardFind("%s/##[TYPE=Clock]"%self.root)
        return self.clocks

    def getChemicalEntities(self, **kwargs):
        """Get the following chemical entities:
        ZombiePool 
        ZombieEnz
        ZombieReac
        """
        self.chemEntities = _moose.wildcardFind("%s/##[TYPE=ZombiePool]"%self.root)
        self.chemEntities += _moose.wildcardFind("%s/##[TYPE=ZombieEnz]"%self.root)
        self.chemEntities += _moose.wildcardFind("%s/##[TYPE=ZombieReac]"%self.root)
        return self.chemEntities

    def getMsgs(self, **kwargs):
        """Get all messages in MOOSE"""
        self.msgs['SingleMsg'] = _moose.wildcardFind('%s/##[TYPE=SingleMsg]'%self.root)
        self.msgs['OneToAllMsg'] = _moose.wildcardFind('%s/##[TYPE=OneToAllMsg]'%self.root)
        return self.msgs

    def populateStoreHouse(self, **kwargs):
        """ Populate all data-structures related with Compartments, Tables, and
        pulse generators.
        """
        if self.filled:
            print_utils.dump("INFO", "Moose elements are already acquired")
            return 
        print_utils.dump("INFO", "Getting moose-datastructure for backend.")
        self.root = kwargs.get('root', '/')
        self.getComparments(**kwargs)
        self.getChemicalEntities(**kwargs)
        self.getTables(**kwargs)
        self.getPulseGens(**kwargs)
        self.getSynChans(**kwargs)
        self.getMsgs(**kwargs)
        self.getClocks()
        self.filled = True

    def clusterNodes(self):
        """Cluster all compartments according to parent path and return them in
        a dictionary """
        population = defaultdict(set)
        for c in self.compartments:
            path = c.path
            parentPath = '/'.join(path.split('/')[0:-1])
            population[parentPath].add(path)
            # Get their channels into cluster as well.
            for channel in c.neighbors['channel']:
                population[parentPath].add(channel.path)
        return population



##
# @brief This is a global object. Import it and call populateStoreHouse() only
# if it has not be called before.
moose_elems = Backend()
