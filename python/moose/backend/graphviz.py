#!/usr/bin/env python

"""graph_utils.py: Graph related utilties. It does not require networkx library.
It writes files to be used with graphviz.

Last modified: Wed Jun 18, 2014  02:20PM

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
from .. import _moose
from .. import print_utils
import inspect
import re
import backend

pathPat = re.compile(r'.+?\[\d+\]$')

##
# @brief Write a graphviz file.
class DotFile():

    def __init__(self):
        self.dot = []
        self.ignorePath = re.compile(r'')

    def setIgnorePat(self, ignorePat):
        self.ignorePat = ignorePat

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
        lhs = self.fix(compA.path)
        rhs = self.fix(compB.path)
        nodeOption = "shape={},label={}".format("box3d", self.label(lhs))
        self.add('\t"{}"[{}];'.format(lhs, nodeOption))

        nodeOption = "shape={},label={}".format("box3d", self.label(rhs))
        self.add('\t"{}"[{}];'.format(rhs, nodeOption))
        self.add('\t"{}" -> "{}";'.format(rhs, lhs))

    def addAxial(self, compA, compB):
        """Add compA and compB axially. """
        self.addRAxial(compB, compA)

    def addLonelyCompartment(self, c):
        """Add a compartment which has not Axial and RAxial connection """
        p = self.fix(c.path)
        nodeOption = "shape={},label={}".format("box3d", self.label(p))
        self.add('\t"{}"[{},color=blue];'.format(p, nodeOption))

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
                           edgeLabel = "color=red,label=synapse,arrowhead=dot"
                           self.add('"{}" -> "{}" [{}];'.format(
                               self.fix(c.path), self.fix(vmSource.path), edgeLabel)
                               )

    def addPulseGen(self, pulseGen, compartments):
        """Add a pulse-generator to dotfile """
        nodeName = self.fix(pulseGen.path)
        nodeOption = "shape=invtriangle,label={}".format(self.label(nodeName))
        self.add('\t"{}"[{}];'.format(nodeName, nodeOption))
        lines = [ '\t"{}" -> "{}"[color=red,label=pulse]'.format(
                        self.fix(pulseGen.path)
                        , self.fix(c.path)) for c in compartments
                        ]
        [self.add(l) for l in lines]

    def addTable(self, table, sources):
        """Add sources to table """
        nodeName = self.fix(table.path)
        nodeOption = "shape=folder,label={}".format(self.label(nodeName))
        self.add('\t"{}"[{}];'.format(nodeName, nodeOption))
        lines = [ '\t"{}" -> "{}"[label=table,color=blue]'.format(
                    self.fix(s.path), nodeName
                    ) for s in sources 
                 ]
        [self.add(l) for l in lines]


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
    paths = []
    comps = []
    obj = _moose.Neutral(obj.path)
    paths.append(obj)
    while len(paths) > 0:
        source = _moose.Neutral(paths.pop())
        targets = source.neighbors['output']
        for t in targets:
            t = _moose.Neutral(t)
            if "moose.Compartment" in t.__str__():
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

    b = backend.Backend()
    b.populateStoreHouse()

    if ignore:
        ignorePat = re.compile(r'%s'%ignore, re.I)
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
            [dotFile.addAxial(c, n) for n in c.neighbors['raxial']]
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
    return True

