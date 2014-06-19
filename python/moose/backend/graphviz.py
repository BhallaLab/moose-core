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

    b = backend.Backend()
    b.populateStoreHouse()

    ignorePat = re.compile(r'')
    if ignore:
        ignorePat = re.compile(r'%s'%ignore, re.I)

    def fix(path):
        '''Fix a given path so it can be written to a graphviz file'''
        # If no [0] is at end of the path then append it.
        global pathPat
        if not pathPat.match(path):
            path = path + '[0]'
        return path

    def label(text, length=4):
        """Create label for a node """
        text = text.replace("[", '').replace("]", '').replace('_', '')
        text = text.replace('/', '')
        return text[-length:]

    compList = b.filterPaths(b.compartments, ignorePat)

    if not compList:
        print_utils.dump("WARN"
                , "No compartment found"
                , frame = inspect.currentframe()
                )
        return None

    dot = set()
    header = "digraph mooseG{"
    header += "\n\tconcentrate=true;\n"
    for c in compList:
        # Each compartment has neighbours connected by axial resistance.
        if c.neighbors['raxial']:
            for n in c.neighbors['raxial']:
                lhs = fix(c.path)
                rhs = fix(n.path)
                nodeOption = "shape={},label={}".format("box3d", label(lhs))
                dot.add('\t"{}"[{}];'.format(lhs, nodeOption))

                nodeOption = "shape={},label={}".format("box3d", label(rhs))
                dot.add('\t"{}"[{}];'.format(rhs, nodeOption))
                dot.add('\t"{}" -> "{}";'.format(rhs, lhs))
        elif c.neighbors['axial']:
            for n in c.neighbors['axial']:
                lhs = fix(c.path)
                rhs = fix(n.path)
                nodeOption = "shape={},label={}".format("box3d", label(lhs))
                dot.add('\t"{}"[{}];'.format(lhs, nodeOption))

                nodeOption = "shape={},label={}".format("box3d", label(rhs))
                dot.add('\t"{}"[{}];'.format(rhs, nodeOption))
                dot.add('\t"{}" -> "{}" [dir=back];'.format(rhs, lhs))
        else:
            p = fix(c.path)
            nodeOption = "shape={},label={}".format("box3d", label(p))
            dot.add('\t"{}"[{},color=blue];'.format(p, nodeOption))

        # Each comparment might also have a synapse on it.
        if c.neighbors['channel']:
            for chan in c.neighbors['channel']:
                print chan

    # Now add the pulse-gen 
    pulseGens = b.pulseGens
    for p in pulseGens:
        comps = getConnectedCompartments(p)
        nodeName = fix(p.path)
        nodeOption = "shape=invtriangle,label={}".format(label(nodeName))
        dot.add('\t"{}"[{}];'.format(nodeName, nodeOption))
        lines = [ '\t"{}" -> "{}"[color=red,label=pulse]'.format(fix(p.path)
                , fix(c.path)) for c in comps 
                ]
        [dot.add(l) for l in lines]

    # Now add tables
    tables = b.tables 
    for t in tables:
        nodeName = fix(t.path)
        nodeOption = "shape=folder,label={}".format(label(nodeName))
        dot.add('\t"{}"[{}];'.format(nodeName, nodeOption))
        sources = t.neighbors['requestOut']
        lines += [ '\t"{}" -> "{}"[label=table,color=blue]'.format(fix(s.path)
                , nodeName) for s in sources 
                ]
        [dot.add(l) for l in lines]

    # Filter all lines which matches the ignorePat 
    dot = '\n'.join(dot)

    dotFile = header + "\n" + dot + "\n}"
    if not filename:
        print(dotFile)
    else:
        with open(filename, 'w') as graphviz:
            print_utils.dump("GRAPHVIZ"
                    , [ "Writing compartment topology to file {}".format(filename)
                        , "Ignoring pattern : {}".format(ignorePat.pattern)
                        ]
                    )
            graphviz.write(dotFile)
    return True

