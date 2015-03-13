#!/usr/bin/env python

"""graph_utils.py: Graph related utilties. It does not require networkx library.
It writes files to be used with graphviz.

Last modified: Fri Mar 13, 2015  06:08PM

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
from .. import _moose as moose
from .. import print_utils
import inspect
import re
import backend
from collections import defaultdict

pathPat = re.compile(r'.+?\[\d+\]$')

def dictToString(params):
    """Convert a dictionary to a valid graphviz option line """
    txt = []
    for k in params:
        txt.append('{}="{}"'.format(k, params[k]))
    return ','.join(txt)

def textToColor(text):
    maxVal = 16777215
    textVal = hex(sum([1677*ord(x) for x in text]))
    textVal = '#{}'.format(textVal[2:])
    return textVal

def short_label(label):
    assert type(label) == str
    label = label.split('/')[-1]
    label = re.sub(r'\[\d+\]', '', label)
    return label

##
# @brief Write a graphviz file.
class DotFile():

    def __init__(self):
        self.dot = []
        self.ignorePat = re.compile(r'')
        self.compShape = "box"
        self.tableShape = "folder"
        self.pulseShape = "invtriangle"
        self.chemShape = "cds"
        self.channelShape = 'doublecircle'
        self.synChanShape = 'Mcircle'
        self.synapseShape = 'star'
        self.bufpoolShape = 'egg'
        self.enzShape = 'star'
        self.reacShape = 'rect'
        self.defaultNodeShape = 'point'
        self.nodes = set()
        # Keep nodes belonging to a subgraph/cluster.
        self.clusters = defaultdict(set)
        self.verbosity = 0
        self.textDict = defaultdict(list)
        # Map each node to a subgraph.
        self.subgraphDict = {}
        self.__total_synapses__ = 0

    def setVerbosity(self, verbosity):
        self.verbosity = verbosity

    def setIgnorePat(self, ignorePat):
        self.ignorePat = ignorePat

    def addHHCHannelNode(self, node, params):
        params['shape'] = self.channelShape
        params['fixedsize'] = 'true'
        params['width'] = 0.1
        params['label'] = ''
        params['height'] = 0.1
        if node.name == 'KConductance':
            params['color'] = 'red'
        elif node.name == 'NaConductance':
            params['color'] = 'green'
        elif node.name == 'CaConductance':
            params['color'] = 'brown'
        else:
            pass
        #params['color'] = textToColor(nodeName)
        return params

    def addNode(self, node, **kwargs):
        """Return a line of dot for given node """
        params = {}
        nodeName = ''
        if type(node) != str:
            nodeName = node.path
            kwargs['node_type'] = type(node)
        else:
            nodeName = node
        nodeName = self.fix(nodeName)
        for k in kwargs:
            if k == 'node_type':
                typeNode = kwargs[k]
                if typeNode == moose.Compartment:
                    params['shape'] = self.compShape
                    params['label'] = short_label(nodeName)
                elif typeNode == moose.HHChannel:
                    params = self.addHHCHannelNode(node, params)
                elif typeNode == moose.SimpleSynHandler:
                    params['shape'] = self.synapseShape
                elif typeNode == moose.SynChan:
                    params['shape'] = self.synChanShape
                    params['label'] = ''
                elif typeNode == moose.ZombiePool:
                    params['shape'] = self.bufpoolShape
                    params['label'] = short_label(nodeName)
                elif typeNode == moose.ZombieEnz:
                    params['shape'] = self.enzShape
                    params['label'] = short_label(nodeName)
                elif typeNode == moose.ZombieReac:
                    params['shape'] = self.reacShape
                    params['label'] = short_label(nodeName)
                else:
                    params['shape'] = self.defaultNodeShape
                    pass
            else:
                params[k] = kwargs[k]
        if 'label' not in params.keys():
            params['label'] = self.label(nodeName)
        nodeText = '"{}" [{}];'.format(nodeName, dictToString(params))
        self.textDict['nodes'].append(nodeText)
        self.nodes.add(nodeName)
        return nodeText

    def addEdge(self, elem1, elem2, **kwargs):
        """Add an edge line to graphviz file """
        node1 = self.fix(elem1.path)
        node2 = self.fix(elem2.path)
        if type(elem1) == type(elem2):
            kwargs['dir'] = 'none'
        elif type(elem1) == moose.HHChannel or type(elem2) == moose.HHChannel:
            kwargs['len'] = 0.1
            kwargs['weight'] = 10
            kwargs['dir'] = 'none'

        elif type(elem1) == moose.SimpleSynHandler or type(elem2) == moose.SimpleSynHandler:
            pass
        elif type(elem1) == moose.SynChan or type(elem2) == moose.SynChan:
            pass
        elif type(elem1) == moose.Synapse or type(elem2) == moose.Synapse:
            #kwargs['len'] = 0.2
            pass
        else:
            kwargs['len'] = 0.2
            kwargs['weight'] = 100
            kwargs['dir'] = 'none'

        # Between objects of same type, do not add multiple edges.
        txt = '"{}" -> "{}" [{}];'.format(node1, node2, dictToString(kwargs))
        if type(elem1) == type(elem2):
            if txt not in self.textDict['edges']:
                self.textDict['edges'].append(txt)
        else:
            self.textDict['edges'].append(txt)
        return txt

    def add(self, line):
        # Add a line to dot 
        if line in self.dot:
            return
        self.dot.append(line)

    def fix(self, path):
        '''Fix a given path so it can be written to a graphviz file'''
        # If no [0] is at end of the path then append it.
        global pathPat
        if not pathPat.match(path):
            path = path + '[0]'
        return path

    def label(self, text):
        """Create label for a node """
        if self.verbosity == 0:
            return ''
        else:
            length = self.verbosity
        text = text.replace("[", '').replace("]", '').replace('_', '')
        text = text.replace('/', '')
        return text[-length:]

    def addSynHandler(self, synHandler, tgt):
        """Add a synaptic handler to network """
        for synapse in synHandler.synapse:
            self.addSynapse(synapse, tgt)
        # We don't have to add this to graphviz. 
        #self.addNode(synHandler.path, node_type=type(synHandler))
        #self.addEdge(synHandler, tgt)

    def addSynapse(self, synapse, post):
        """Get a synapse and add an edge from its pre-synaptic terminal """
        for spikeGen in synapse.neighbors['addSpike']:
            for sg in spikeGen:
                for pre in sg.neighbors['Vm']:
                    self.__total_synapses__ += 1
                    self.addEdge(pre, post, arrowhead='box')

    def addChemElement(self, chemNode):
        """Add a given moose.element representing chemical element to dot
        file"""
        self.addNode(chemNode)


    def writeDotFile(self, fileName, options):
        """Write the topology to a graphviz file 

        nodes_shape : If this parameter is passed, all nodes shapes is set to
        this value. Useful when large number of nodes are present in graph.
        Automatically set to "point" if no of nodes in graph is more than 200.

        ignore: If parameter ignores all nodes which have any of these words in
        them.
        """
        dotText = ''
        dotText += '\n'.join(self.textDict['header'])
        dotText += "\n\n"

        clusters = self.textDict['subgraphs']
        for cname in clusters:
            clusterText = 'subgraph cluster_%s{\n' % cname
            for n in clusters[cname]:
                clusterText += '"%s"[label="",shape="none"];\n' % self.fix(n)
            clusterText += '}\n'
            dotText += clusterText

        dotText += '\n'.join(self.textDict['nodes'])
        dotText += '\n'.join(self.textDict['edges'])
        dotText =  dotText + "\n}"

        # If nodes_shape parameter is given, then draw all nodes with this
        # shape.
        globalNodeShape = options.get('nodes_shape', None)
        if globalNodeShape:
            dotText = re.sub(r'shape=\"\w+\"'
                        , 'shape="%s"'%globalNodeShape
                        , dotText
                        )

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
    obj = moose.Neutral(obj.path)
    paths.append(obj)
    while len(paths) > 0:
        source =  paths.pop()
        targets = source.neighbors['output']
        for t in targets:
            t = moose.Neutral(t)
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
# @param root If set only this pat is searched, else every electrical and
# chemical compartment is searched and dumped to graphviz file.
# @param cluster. When set to true, cluster nodes together as per the root of their
# path e.g. /library/cell1 and /library/cell2 are grouped together because they
# are both under /library path.
# @param Ignore paths containing this pattern. A regular expression. 
# @param add_recorders (False). If true add tables to graph as well.
#
# @return None. 
def writeGraphviz(filename=None, root='/', cluster=True, ignore=None, **kwargs):
    '''This is  a generic function. It takes the the pattern, search for paths
    and write a graphviz file.
    '''

    global dotFile
    verbose = kwargs.get('verbose', 0)
    dotFile.setVerbosity(verbose)

    print_utils.dump("GRAPHVIZ"
            , "Preparing graphviz file for writing"
            )

    if not backend.moose_elems.filled:
        backend.moose_elems.populateStoreHouse(root=root)

    b = backend.moose_elems
    ignorePat = re.compile(r'^abcd')
    if ignore:
        ignorePat = re.compile(r'%s' % ignore, re.I)
        dotFile.setIgnorePat(ignorePat)

    chemList = b.filterPaths(b.chemEntities, ignorePat)
    [ dotFile.addChemElement(c) for c in chemList ]

    header = ["digraph mooseG{"]
    header.append('\ngraph[];')
    dotFile.textDict['header'].append('\n'.join(header))
    
    for sm in b.msgs['SingleMsg']:
        srcs, tgts = b.filterPaths(sm.e1, ignorePat), b.filterPaths(sm.e2, ignorePat)
        for i, src in enumerate(srcs):
            tgt = tgts[i]
            if type(src) == moose.SimpleSynHandler:
                dotFile.addSynHandler(src, tgt)
            elif type(src) == moose.Table or type(tgt) == moose.Table:
                pass
            elif type(src) == moose.PulseGen or type(tgt) == moose.PulseGen:
                pass
            elif type(src) == moose.SpikeGen or type(tgt) == moose.SpikeGen:
                pass
            else:
                dotFile.addNode(src)
                dotFile.addNode(tgt)
                dotFile.addEdge(src, tgt)
    print_utils.dump("TODO", "OneToAllMsgs are not added to graphviz file")
            
    # If cluster is True then cluster the compartments together. Also create a
    # dictionary from which we can fetch the clustername of a compartment.
    # NOTE: THIS MUST BE THE LAST STEP BEFORE WRITING TO GRAPHVIZ FILE.
    subgraphDict = {}
    if cluster:
        pops = b.clusterNodes()
        for pop in pops:
            clustername = pop.translate(None, '/[]')
            subgraphDict[clustername] = pops[pop]
            for x in pops[pop]: dotFile.subgraphDict[x] = clustername
    dotFile.textDict['subgraphs'] = subgraphDict
    dotFile.writeDotFile(filename, kwargs)
    return True

