#!/usr/bin/env python

"""plot_utils.py: Some utility function for plotting data in moose.

Last modified: Mon May 26, 2014  10:18AM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import pylab 
import _moose
import print_utils as debug

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

def xyToString( yvec, xvec, sepby = ' '):
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
    textLines = xyToString( yvec, xvec)
    with open(file, "w") as dataF:
        dataF.write(textLines)

def saveAsGnuplot( yvec, xvec, file):
    ''' Save the plot as stand-alone gnuplot script '''
    if file is None:
        return
    print("[INFO] Saving plot data to a gnuplot-script: {}".format(file))
    dataText = xyToString( yvec, xvec )
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

def scaleVector(vec, scaleF):
    """ Scale a vector by a factor """
    if scaleF == 1.0 or scaleF is None:
        return vec
    else:
        return [ x*scaleF for x in vec ]

def scaleAxis(xvec, yvec, scaleX, scaleY):
    """ Multiply each elements by factor """
    xvec = scaleVector( xvec, scaleX )
    yvec = scaleVector( yvec, scaleY )
    return xvec, yvec

def reformatTable(table, kwargs):
    """ Given a table return x and y vectors with proper scaling """
    if type(table) == _moose.Table:
        vecY = table.vector 
        vecX = range(len(vecY))
    elif type(table) == tuple:
        vecX, vecY = table
    xscale = kwargs.get('xscale', 1.0)
    yscale = kwargs.get('yscale', 1.0)
    return scaleAxis(vecX, vecY, xscale, yscale)

def plotTable(table, standalone=True, file=None, **kwargs):
    """Plot a given table. It plots table.vector

    This function can scale the x-axis. By default, y-axis and x-axis scaling is
    done by a factor of 1.  

    Pass 'xscale' and/or 'yscale' argument to function to modify scales.
    
    """
    if not type(table) == _moose.Table:
        msg = "Expected moose.Table, got {}".format( type(table) )
        raise TypeError(msg)
    if standalone:
        pylab.figure()

    vecX, vecY = reformatTable(table, kwargs)
    pylab.plot(vecX, vecY)
    if file and standalone:
        debug.dump("PLOT", "Saving plot to {}".format(file))
        pylab.savefig(file)
    elif standalone:
        pylab.show()

def plotTables(tables, file=None, **kwargs):
    """Plot a list of tables onto one figure only.
    """
    assert type(tables) == list, "Expected a list of moose.Tables"
    for t in tables:
        plotTable(t, standalone = False, file = None, **kwargs)
    if file:
        debug.dump("PLOT", "Saving plots to file {}".format(file))
        try:
            pylab.savefig(file)
        except Exception as e:
            debug.dump("WARN"
                    , "Failed to save figure, plotting onto a window"
                    )
            pylab.show()
    else:
        pylab.show()

def saveTables(tables, file=None, **kwargs):
    """Save a list to tables to a data file. """
    assert type(tables) == list, "Expecting a list of moose.Table"
    plots = []
    xaxis = None
    for t in tables:
        vecX, vecY = reformatTable(t, kwargs)
        plots.append(vecY)
        if xaxis:
            if xaxis != vecX:
                raise UserWarning("Tables must have same x-axis")
        else:
            xaxis = vecX
        tableText = ""
        for i, x in enumerate(xaxis):
            tableText += "{} ".format(x)
            tableText += " ".join(['%s'%p[i] for p in plots])
            tableText += "\n"
    if file is None:
        print(tableText)
    else:
        debug.dump("PLOT", "Saving tables data to file {}".format(file))
        with open(file, "w") as f:
            f.write(tableText)
    
   
