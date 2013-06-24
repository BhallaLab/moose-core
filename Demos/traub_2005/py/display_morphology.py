# display_morphology.py --- 
# 
# Filename: display_morphology.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Mar  8 11:26:13 2013 (+0530)
# Version: 
# Last-Updated: Mon Jun 24 18:11:26 2013 (+0530)
#           By: subha
#     Update #: 309
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Draw the schematic diagram of cells using networkx
# 
# 

# Change log:
# 
# 
# 
# 

# Code:

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import moose
import cells

def node_sizes(g):
    """Calculate the 2D projection area of each compartment.
    
    g: graph whose nodes are moose Compartment objects.
    
    return a numpy array with compartment areas in 2D projection
    normalized by the maximum.

    """
    sizes = []
    comps = [moose.Compartment(n) for n in g.nodes()]
    sizes = np.array([c.length * c.diameter for c in comps])
    soma_i = [ii for ii in range(len(comps)) if comps[ii].path.endswith('comp_1')]
    sizes[soma_i] *= np.pi/4 # for soma, length=diameter. So area is dimater^2 * pi / 4
    return sizes / max(sizes)
    
def cell_to_graph(cell):
    """Convert a MOOSE compartmental neuron into a graph describing
    the topology of the compartments

    """        
    es = [(c1.path, c2.path, {'weight': 2/ (moose.Compartment(c1).Ra + moose.Compartment(c2).Ra)}) \
              for c1 in moose.wildcardFind('%s/##[ISA=Compartment]' % (cell.path)) \
              for c2 in moose.Compartment(c1).neighbours['raxial']]
    g = nx.Graph()
    g.add_edges_from(es)
    return g

def axon_dendrites(g):
    """Get a 2-tuple with list of nodes representing axon and list of
    nodes representing dendrites.

    g: graph whose nodes are compartments 

    """
    axon = []
    soma_dendrites = []
    for n in g.nodes():
        if moose.exists('%s/CaPool' % (n)):
            soma_dendrites.append(n)
        else:
            axon.append(n)
    return (axon, soma_dendrites)

def plot_cell_topology(cell):
    g = cell_to_graph(cell)
    axon, sd = axon_dendrites(g)
    node_size = node_sizes(g)
    weights = np.array([g.edge[e[0]][e[1]]['weight'] for e in g.edges()])
    # print min(weights), max(weights)
    pos = nx.graphviz_layout(g,prog='twopi',root=cell.path + '/comp_1')
    # pos = nx.spring_layout(g)
    nx.draw_networkx_edges(g, pos, width=10*weights/max(weights), edge_color='gray', alpha=0.8)
    nx.draw_networkx_nodes(g, pos, with_labels=False, node_size=node_size * 500, node_color=map(lambda x: 'k' if x in axon else 'gray', g.nodes()), linewidths=[1 if n.endswith('comp_1') else 0 for n in g.nodes()], alpha=0.8)
    plt.title(cell.__class__.__name__)

from matplotlib.backends.backend_pdf import PdfPages

import sys
from getopt import getopt

if __name__ == '__main__':
    print sys.argv
    optlist, args = getopt(sys.argv[0], 'hpc:', ['help'])
    celltype = ''
    pdf = ''
    for arg in optlist:
        if arg[0] == '-c':
            celltype = arg[1]
        elif arg[0] == '-p':
            pdf = arg[1]
        elif arg[0] == '-h' or arg[0] == '--help':
            print 'Usage: %s [-c CellType [-p filename]]' % (sys.argv[0])
            print 'Display/save the morphology of cell type "CellType".'
            print 'Options:'
            print '-c celltype (optional) display only an instance of the specified cell type. If CellType is empty or not specified, all prototype cells are displayed.'
            print '-p  filename (optional) save outputin a pdf file named "filename".'
            print '-h,--help print this help'
            sys.exit(0)
    figures = []
    if len(celltype) > 0:
        try:
            fig = plt.figure()
            figures.append(fig)
            cell = cells.init_prototypes()[celtype]
            plot_cell_topology(cell)
        except KeyError:
            print '%s: no such cell type. Available are:' % (celltype)
            for ii in cells.init_prototypes().keys():
                print ii,
            print 
            sys.exit(1)    
    else:
        for cell, proto in cells.init_prototypes().items():
            figures.append(plt.figure())
            plot_cell_topology(proto)
    plt.axis('off')
    if len(pdf) > 0:
        pdfout = PdfPages(pdf)
        for fig in figures:
            pdfout.savefig(fig)
        pdfout.close()
    else:
        plt.show()

# 
# display_morphology.py ends here
