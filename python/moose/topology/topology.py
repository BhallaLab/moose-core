"""topology.py: 

    Store the MOOSE network as a graph.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


import networkx as nx
import matplotlib.pyplot as plt
from .. import _moose as moose
from .. import print_utils as pu

class Network(object):

    def __init__(self, filename):
        self.graph = nx.MultiDiGraph()
        self.outfileName = filename

    def addEdge(self, src, tgt):
        srcPath = src.path
        tgtPath = tgt.path
        self.graph.add_node(srcPath, type=type(src))
        self.graph.add_node(tgtPath, type=type(tgt))


    def addMessage(self, m):
        e1, e2 = m.e1, m.e2
        for i, e in enumerate(e1):
            src, tgt = e, e2[i]
            self.addEdge(src, tgt)

    def build(self, **kwargs):
        """Build the network """
        msgs = moose.wildcardFind('/##[TYPE=SingleMsg]')
        for m in msgs:
            self.addMessage(m)

def writeNetwork(filename=None, **kwargs):
    pu.info("Writing network to %s" % filename)
    network = Network(filename)
    network.build()
    if not filename:
        nx.draw(network.graph)
        #plt.show()
