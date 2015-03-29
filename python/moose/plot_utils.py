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

import matplotlib.pyplot as plt
import _moose as moose
import print_utils as pu 
import numpy as np

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
    if type(table) == moose.Table:
        vecY = table.vector 
        vecX = np.arange(len(vecY))
    elif type(table) == tuple:
        vecX, vecY = table
    xscale = kwargs.get('xscale', 1.0)
    yscale = kwargs.get('yscale', 1.0)
    return scaleAxis(vecX, vecY, xscale, yscale)

def plotTable(table, **kwargs):
    """Plot a given table. It plots table.vector

    This function can scale the x-axis. By default, y-axis and x-axis scaling is
    done by a factor of 1.  

    Pass 'xscale' and/or 'yscale' argument to function to modify scales.
    
    """
    if not type(table) == moose.Table:
        msg = "Expected moose.Table, got {}".format( type(table) )
        raise TypeError(msg)

    vecX, vecY = reformatTable(table, kwargs)
    plt.plot(vecX, vecY, label = kwargs.get('label', ""))
    plt.legend(loc='best', framealpha=0.4)

def plotTables(tables, outfile=None, **kwargs):
    """Plot a list of tables onto one figure only.
    """
    assert type(tables) == dict, "Expected a dict of moose.Table"
    plt.figure(figsize=(10, 1.5*len(tables)))
    subplot = kwargs.get('subplot', True)
    for i, tname in enumerate(tables):
        if subplot:
            plt.subplot(len(tables), 1, i)
        yvec = tables[tname].vector 
        xvec = np.linspace(0, moose.Clock('/clock').currentTime, len(yvec))
        plt.plot(xvec, yvec, label=tname)
        plt.legend(loc='best', framealpha=0.4)
    
    plt.tight_layout()
    if outfile:
        pu.dump("PLOT", "Saving plots to file {}".format(outfile))
        try:
            plt.savefig(outfile)
        except Exception as e:
            pu.dump("WARN"
                    , "Failed to save figure, plotting onto a window"
                    )
            plt.show()
    else:
        plt.show()

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
        pu.dump("PLOT", "Saving tables data to file {}".format(file))
        with open(file, "w") as f:
            f.write(tableText)
    
   

def plotVector(vec, xvec = None, **options):
    """plotVector: Plot a given vector. On x-axis, plot the time.

    :param vec: Given vector.
    :param **kwargs: Optional to pass to maplotlib.
    """

    assert type(vec) == np.ndarray, "Expected type %s" % type(vec)

    if xvec is None:
        clock = moose.Clock('/clock')
        xx = np.linspace(0, clock.currentTime, len(vec))
    else:
        xx = xvec[:]

    assert len(xx) == len(vec), "Expecting %s got %s" % (len(vec), len(xvec))

    plt.plot(xx, vec, label=options.get('label', ''))

    if xvec is None:
        plt.xlabel = 'Time (sec)'
    else:
        plt.xlabel = options.get('xlabel', '')
    
    plt.ylabel = options.get('ylabel', '')
    plt.title = options.get('title', '')

    if(options.get('legend', True)):
        plt.legend(loc='best', framealpha=0.4, prop={'size' : 6})


def saveRecords(dataDict, xvec = None, **kwargs):
    """saveRecords Given a dictionary of data with (key, vector) pair, it saves
    them.

    :param dataDict:
    :param **kwargs:
    """

    assert type(dataDict) == dict, "Got %s" % type(dataDict)

    outfile = kwargs.get('outfile', 'data.moose')

    filters = [ x.lower() for x in kwargs.get('filter', [])]
    pu.info("Writing data to %s" % outfile)
    with open(outfile, 'w') as f:
        for k in dataDict:
            yvec = dataDict[k].vector
            if xvec is None:
                clock = moose.Clock('/clock')
                xx = np.linspace(0, clock.currentTime, len(yvec))
            else:
                xx = xvec[:]
            xline = ','.join([str(x) for x in xx])
            yline = ','.join([str(y) for y in yvec])
            f.write('"%s:x",%s\n' % (k, xline))
            f.write('"%s:y",%s\n' % (k, yline))
    pu.info(" .. Done writing data to moose-data file")

def plotRecords(dataDict, xvec = None, **kwargs):
    """plotRecords Plot given records in dictionary.

    :param dataDict:
    :param xvec: If None, use moose.Clock to generate xvec.
    :param **kwargs:
    """

    legend = kwargs.get('legend', True)
    outfile = kwargs.get('outfile', None)
    subplot = kwargs.get('subplot', False)
    filters = [ x.lower() for x in kwargs.get('filter', [])]

    plt.figure(figsize=(10, 1.5*len(dataDict)))
    for i, k in enumerate(dataDict):
        pu.info("+ Plotting for %s" % k)
        plotThis = False
        if not filters: plotThis = True
        for accept in filters:
            if accept in k.lower(): 
                plotThis = True
                break
                
        if plotThis:
            if not subplot: 
                yvec = dataDict[k].vector
                plotVector(yvec, xvec, **kwargs)
            else:
                plt.subplot(len(dataDict), 1, i)
                yvec = dataDict[k].vector
                plotVector(yvec, xvec, **kwargs)
    try:
        plt.tight_layout()
    except: pass

    if outfile:
        pu.info("Writing plot to %s" % outfile)
        plt.savefig("%s" % outfile)
    else:
        plt.show()
