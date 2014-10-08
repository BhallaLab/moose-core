#!/usr/bin/env python

"""graph_utils.py: Graph related utilties. It does not require networkx library.
It writes files to be used with graphviz.

Last modified: Tue Jul 15, 2014  12:21PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
from .. import _moose
from .. import print_utils
import inspect
import re
import backend

pathPat = re.compile(r'.+?\[\d+\]$')

def dictToString(**kwargs):
    """Convert a dictionary to a valid graphviz option line """
    txt = []
    for k in kwargs:
        txt.append('{}="{}"'.format(k, kwargs[k]))
    return ','.join(txt)

##
# @brief Write a graphviz file.
class DotFile():

    def __init__(self):
        self.dot = []
        self.ignorePat = re.compile(r'')
        self.compShape = "box3d"
        self.tableShape = "folder"
        self.pulseShape = "invtriangle"

    def setIgnorePat(self, ignorePat):
        self.ignorePat = ignorePat

    def addNode(self, nodeName, **kwargs):
        """Return a line of dot for given node """
        nodeName = self.fix(nodeName)
        if 'label' not in kwargs.keys():
            kwargs['label'] = self.label(nodeName)
        nodeText = '"{}" [{}];'.format(nodeName, dictToString(**kwargs))
        self.add(nodeText)
        return nodeText

    def addEdge(self, node1, node2, **kwargs):
        """Add an edge line to graphviz file """
        node1 = self.fix(node1)
        node2 = self.fix(node2)
        txt = '"{}" -> "{}" [{}];'.format(node1, node2, dictToString(**kwargs))
        self.add(txt)
        return txt

    def add(self, line):
        # Add a line to dot 
        if line in self.dot:
            return
        self.dot.append(line)

    def writeDotFile(self, fileName):
        dotText = '\n'.join(self.dot)
        dotText =  dotText + "\n}"
        if not fileName:
            print(dotText)
            return

        with open(fileName, 'w') as graphviz:
            print_utils.dump("GRAPHVIZ"
                    , [ "Writing compartment topology to file {}".format(fileName)
                        , "Ignoring pattern : {}".format(self.ignorePat.pattern)
                        ]
                    )
            graphviz.write(dotText)

    def fix(self, path):
        '''Fix a given path so it can be written to a graphviz file'''
        # If no [0] is at end of the path then append it.
        global pathPat
        if not pathPat.match(path):
            path = path + '[0]'
        return path

    def label(self, text, length=4):
        """Create label for a node """
        text = text.replace("[", '').replace("]", '').replace('_', '')
        text = text.replace('/', '')
        return text[-length:]

    def addRAxial(self, compA, compB):
        """Connect compA with compB """
        lhs = compA.path
        self.addNode(lhs, shape=self.compShape)
        rhs = compB.path
        self.addNode(rhs, shape=self.compShape)
        self.addEdge(rhs, lhs)

    def addAxial(self, compA, compB):
        """Add compA and compB axially. """
        self.addRAxial(compB, compA)

    def addLonelyCompartment(self, compartment):
        """Add a compartment which has not Axial and RAxial connection """
        p = compartment.path
        self.addNode(p, shape=self.compShape, color='blue')

    def addChannel(self, c, chan):
        """Find synapses in channels and add to dot."""
        for sc in chan:
           if "moose.SynChan" not in sc.__str__():
               continue
           for synapse in sc.synapse:
               spikeSources =  synapse.neighbors['addSpike']
               for ss in spikeSources:
                   for s in ss:
                       for vmSource in  s.neighbors['Vm']:
                           self.addEdge(c.path
                                   , vmSource.path
                                   , color = 'red'
                                   , label = 'synapse'
                                   , arrowhead = 'dot'
                                   )

    def addPulseGen(self, pulseGen, compartments):
        """Add a pulse-generator to dotfile """
        nodeName = pulseGen.path
        self.addNode(nodeName, shape=self.pulseShape)
        for c in compartments:
            self.addEdge(pulseGen.path, c.path, color='red', label='pulse') 


    def addTable(self, table, sources):
        """Add sources to table """
        nodeName = table.path
        self.addNode(nodeName, shape=self.tableShape)
        for s in sources:
            self.addEdge(s.path, nodeName, label='table', color='blue')


##
# @brief Object of DotFile.
dotFile = DotFile()

##
# @brief Given a moose object which is not Compartment, return all possible
# paths to Compartments.
#
# @param obj A moose object.
#
# @return 
def getConnectedCompartments(obj):
    """List all compartments connected to this obj."""
    if "moose.Compartment" in obj.__str__():
        return []
    if "moose.ZombieCompartment" in obj.__str__():
        return []

    paths = []
    comps = []
    obj = _moose.Neutral(obj.path)
    paths.append(obj)
    while len(paths) > 0:
        source =  paths.pop()
        targets = source.neighbors['output']
        for t in targets:
            t = _moose.Neutral(t)
            if "moose.Compartment" in t.__str__():
                comps.append(t)
            elif "moose.ZombieCompartment" in t.__str__():
                comps.append(t)
            else:
                paths.append(t)
    return comps
##
# @brief Write a graphviz topology file.
#
# @param filename Name of the output file.
# @param pat Genesis pattern for searching all compratments; default
# '/##[TYPE=Compartment'
# @param Ignore paths containing this pattern. A regular expression. 
#
# @return None. 
def writeGraphviz(filename=None, pat='/##[TYPE=Compartment]', ignore=None):
    '''This is  a generic function. It takes the the pattern, search for paths
    and write a graphviz file.
    '''

    global dotFile

    print_utils.dump("GRAPHVIZ"
            , "Preparing graphviz file for writing"
            )

    if not backend.moose_elems.filled:
        backend.moose_elems.populateStoreHouse()
    b = backend.moose_elems

    ignorePat = re.compile(r'^abcd')
    if ignore:
        ignorePat = re.compile(r'%s' % ignore, re.I)
        dotFile.setIgnorePat(ignorePat)

    compList = b.filterPaths(b.compartments, ignorePat)

    if not compList:
        print_utils.dump("WARN"
                , "No compartment found"
                , frame = inspect.currentframe()
                )
        return None

    header = "digraph mooseG{"
    header += "\n\tconcentrate=true;\n"

    dotFile.add(header)

    for c in compList:
        # Each compartment has neighbours connected by axial resistance.
        if c.neighbors['raxial']:
            [dotFile.addRAxial(c, n) for n in c.neighbors['raxial']]
        elif c.neighbors['axial']:
            [dotFile.addAxial(c, n) for n in c.neighbors['axial']]
        else:
            dotFile.addLonelyCompartment(c)

        # Each comparment might also have a synapse on it.
        chans = c.neighbors['channel']
        [dotFile.addChannel(c, chan) for chan in chans]

    # Now add the pulse-gen 
    pulseGens = b.pulseGens
    for p in pulseGens:
        comps = getConnectedCompartments(p)
        dotFile.addPulseGen(p, comps)
        
    # Now add tables
    tables = b.tables 
    for t in tables:
        sources = t.neighbors['requestOut']
        dotFile.addTable(t, sources)

    dotFile.writeDotFile(filename)
    return True

