#!/usr/bin/env python

"""graph_utils.py: Graph related utilties. It does not require networkx library.
It writes files to be used with graphviz.

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
from .. import _moose
from .. import print_utils
import inspect
import re

pathPat = re.compile(r'.+?\[\d+\]$')

def getMooseCompartments(pat, ignorePat=None):
    ''' Return a list of paths for a given pattern. '''
    def ignore(x):
        if ignorePat.search(x.path):
            return False
        return True

    if type(pat) != str:
        pat = pat.path
        assert type(pat) == str

    moose_paths = _moose.wildcardFind(pat)
    if ignorePat:
        moose_paths = filter(ignore, moose_paths)
    return moose_paths

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

        
    compList = getMooseCompartments(pat, ignorePat)
    if not compList:
        print_utils.dump("WARN"
                , "No compartment found"
                , frame = inspect.currentframe()
                )
        return None

    dot = []
    dot.append("digraph G {")
    dot.append("\tconcentrate=true;")
    for c in compList:
        if c.neighbors['raxial']:
            for n in c.neighbors['raxial']:
                lhs = fix(c.path)
                rhs = fix(n.path)
                dot.append('\t"{}" -> "{}";'.format(lhs, rhs))
        elif c.neighbors['axial']:
            for n in c.neighbors['axial']:
                lhs = fix(c.path)
                rhs = fix(n.path)
                dot.append('\t"{}" -> "{}" [dir=back];'.format(lhs, rhs))
        else:
            p = fix(c.path)
            dot.append('\t"{}"'.format(p))
    # Filter all lines which matches the ignorePat 
    dot.append('}')
    dot = '\n'.join(dot)
    if not filename:
        print(dot)
    else:
        with open(filename, 'w') as graphviz:
            print_utils.dump("GRAPHVIZ"
                    , [ "Writing compartment topology to file {}".format(filename)
                        , "Ignoring pattern : {}".format(ignorePat.pattern)
                        ]
                    )
            graphviz.write(dot)
    return True

