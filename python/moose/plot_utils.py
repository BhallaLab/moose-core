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

import pylab 

def plotAscii(yvec, xvec = None, file=None):
    """Plot two list-like object in terminal using gnuplot.
    If file is given then save data to file as well.
    """
    if xvec is None:
        plotInTerminal(yvec, range( len(yvec) ), file=file)
    else:
        plotInTerminal(yvec, xvec, file=file)

def plotInTerminal(yvec, xvec = None, file=None):
    '''
    Plot given vectors in terminal using gnuplot.

    If file is not None then write the data to a file.
    '''
    import subprocess
    g = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
    g.stdin.write("set term dumb 100 25\n")
    g.stdin.write("plot '-' using 1:2 title '{}' with linespoints\n".format(file))
    if file:
        saveAsGnuplot(yvec, xvec, file=file)
    for i,j in zip(xvec, yvec):
        g.stdin.write("%f %f\n" % (i, j))
    g.stdin.write("\n")
    g.stdin.flush()

def listsToString( yvec, xvec, sepby = ' '):
    """ Given two list-like objects, returns a text string. 
    """
    textLines = []
    for y, x in zip( yvec, xvec ):
        textLines.append("{}{}{}".format(y, sepby, x))
    return "\n".join(textLines)


def saveNumpyVec( yvec, xvec, file):
    """save the numpy vectors to a data-file
    
    """
    if file is None:
        return
    print("[INFO] Saving plot data to file {}".format(file))
    textLines = listsToString( yvec, xvec)
    with open(file, "w") as dataF:
        dataF.write(textLines)

def saveAsGnuplot( yvec, xvec, file):
    ''' Save the plot as stand-alone gnuplot script '''
    if file is None:
        return
    print("[INFO] Saving plot data to a gnuplot-script: {}".format(file))
    dataText = listsToString( yvec, xvec )
    text = []
    text.append("#!/bin/bash")
    text.append("gnuplot << EOF")
    text.append("set term post eps")
    text.append("set output \"{0}.eps\"".format(file))
    text.append("plot '-' using 0:1 title '{0}'".format(file))
    text.append(dataText)
    text.append("EOF")
    with open(file+".gnuplot","w") as gnuplotF:
        gnuplotF.write("\n".join(text))


def plotTable(table, subfig=False, file=None, **kwargs):
    """Plot a given table. It plots table.vector

    This function can scale the x-axis. By default, y-axis and x-axis scaling is
    done by a factor of 1.  

    Pass 'xscale' and/or 'yscale' argument to function to modify scales.
    
    """

    if not subfig:
        pylab.figure()
    else:
        pylab.subplot()
    vector = table.vector 
    xscale = kwargs.get('xscale', 1.0)
    yscale = kwargs.get('yscale', 1.0)
    if xscale != 1.0:
        xvector = [ xscale*x for x in range(len(vector)) ]
    if yscale != 1.0:
        yvector = [ yscale * v for v in vector ]
    else:
        yvector = vector
    pylab.plot(xvector, yvector)
    if file:
        print("[UTIL] Saving plot to {}".format(file))
        pylab.savefig(file)
