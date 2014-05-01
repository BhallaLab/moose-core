#!/usr/bin/env python

"""plot_utils.py: Some utility function for plotting data in moose.

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

def plotAscii(yvec, xvec = None):
    """docstring for plotAscii"""
    if xvec is None:
        plotInTerminal(yvec, xvec = range( len(yvec) ))
    else:
        assert type(yvec) == type(xvec)
        plotInTerminal(yvec, xvec)

def plotInTerminal(yvec, xvec = None):
    '''
    Plot given vectors in terminal using gnuplot.
    '''
    import subprocess
    g = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
    g.stdin.write("set term dumb 79 25\n")
    g.stdin.write("plot '-' using 1:2 title 'Line1' with linespoints \n")
    for i,j in zip(xvec, yvec):
        g.stdin.write("%f %f\n" % (i, j))
    g.stdin.write("e\n")
    g.stdin.flush()
