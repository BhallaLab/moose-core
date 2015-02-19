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

    def addNode(self, elem):
        path = elem.path
        self.graph.add_node(path, type=type(elem))
        if type(elem) == moose.Compartment:
            x, y = elem.x + 0.1* elem.z, elem.y + 0.1 * elem.z
            self.graph.node[path]['pos'] = (x, y)
        return path

    def addEdge(self, src, tgt):
        n1 = self.addNode(src)
        n2 = self.addNode(tgt)
        self.graph.add_edge(n1, n2)

    def computeNodePositions(self):
        for n, d in self.graph.nodes(data=True):
            if d['type'] == moose.Compartment:
                print d

    def addMessage(self, m):
        e1, e2 = m.e1, m.e2
        for i, e in enumerate(e1):
            src, tgt = e, e2[i]
            self.addEdge(src, tgt)

    def build(self, **kwargs):
        """Build the network """
        pu.info("TODO: Only SingleMsgs are added")
        msgs = moose.wildcardFind('/##[TYPE=SingleMsg]')
        for m in msgs:
            self.addMessage(m)
        self.computeNodePositions()

def writeNetwork(filename=None, **kwargs):
    pu.info("Writing network to %s" % filename)
    network = Network(filename)
    network.build()
    if not filename:
        print network.graph
        pu.debug("Plotting network")
        #nx.draw_networkx(network.graph, with_labels=False)
        #plt.show()
