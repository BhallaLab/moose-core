# -*- coding: utf-8 -*-
"""test_table_streaming_support.py:

Test the streaming support in moose.Table.

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2016, Dilawar Singh"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import os
import sys
import moose
import numpy as np
print( '[INFO] Using moose form %s' % moose.__file__ )

def print_table( table ):
    msg = ""
    msg += " datafile : %s" % table.datafile
    msg += " useStreamer: %s" % table.useStreamer
    msg += ' Path: %s' % table.path
    print( msg )

def test_small( ):
    moose.CubeMesh( '/compt' )
    r = moose.Reac( '/compt/r' )
    a = moose.Pool( '/compt/a' )
    a.concInit = 1
    b = moose.Pool( '/compt/b' )
    b.concInit = 2
    c = moose.Pool( '/compt/c' )
    c.concInit = 0.5
    moose.connect( r, 'sub', a, 'reac' )
    moose.connect( r, 'prd', b, 'reac' )
    moose.connect( r, 'prd', c, 'reac' )
    r.Kf = 0.1
    r.Kb = 0.01

    tabA = moose.Table2( '/compt/a/tabA' )
    #  tabA.format = 'npy'
    tabA.useStreamer = True   # Setting format alone is not good enough

    # Setting datafile enables streamer.
    tabB = moose.Table2( '/compt/b/tabB' )
    tabB.datafile = 'table2.npy'

    tabC = moose.Table2( '/compt/c/tabC' )
    tabC.datafile = 'tablec.csv'

    moose.connect( tabA, 'requestOut', a, 'getConc' )
    moose.connect( tabB, 'requestOut', b, 'getConc' )
    moose.connect( tabC, 'requestOut', c, 'getConc' )

    moose.reinit( )
    [ print_table( x) for x in [tabA, tabB, tabC] ]
    runtime = 1000
    print( 'Starting moose for %d secs' % runtime )
    moose.start( runtime, 1 )

    # Now read the numpy and csv and check the results.
    a = np.loadtxt( tabA.datafile, skiprows=1 )
    b = np.load( 'table2.npy' )
    c = np.loadtxt( 'tablec.csv', skiprows=1 )
    assert (len(a) == len(b) == len(c))
    print( ' MOOSE is done' )

def buildLargeSystem(useStreamer = False):
    # create a huge system.
    if moose.exists('/comptB'):
        moose.delete('/comptB')
    moose.CubeMesh( '/comptB' )

    tables = []
    for i in range(300):
        r = moose.Reac('/comptB/r%d'%i)
        a = moose.Pool('/comptB/a%d'%i)
        a.concInit = 10.0
        b = moose.Pool('/comptB/b%d'%i) 
        b.concInit = 2.0
        c = moose.Pool('/comptB/c%d'%i)
        c.concInit = 0.5
        moose.connect( r, 'sub', a, 'reac' )
        moose.connect( r, 'prd', b, 'reac' )
        moose.connect( r, 'prd', c, 'reac' )
        r.Kf = 0.1
        r.Kb = 0.01

        # Make table name large enough such that the header is larger than 2^16
        # . Numpy version 1 can't handle such a large header. If format 1 is
        # then this test will fail.
        t = moose.Table2('/comptB/TableO1%d'%i + 'abc'*100)
        moose.connect(t, 'requestOut', a, 'getConc')
        tables.append(t)

    if useStreamer:
        s = moose.Streamer('/comptB/streamer')
        s.datafile = 'data2.npy'
        print("[INFO ] Total tables %d" % len(tables))

        # Add tables using wilcardFind.
        s.addTables(moose.wildcardFind('/comptB/##[TYPE=Table2]'))

        print("Streamer has %d table" % s.numTables)
        assert s.numTables == len(tables), (s.numTables, len(tables))

    moose.reinit()
    moose.start(10)

    if useStreamer:
        # load the data
        data = np.load(s.datafile)
        header = str(data.dtype.names)
        assert len(header) > 2**16
    else:
        data = { x.columnName : x.vector for x in tables }
    return data

def test_large_system():
    # Get data without streamer and with streamer.
    # These two must be the same.
    X = buildLargeSystem(False)   # without streamer
    Y = buildLargeSystem(True)    # with streamer.

    # X has no time.
    assert len(X) == len(Y.dtype.names)-1, (len(X), Y.dtype)

    # same column names.
    xNames = list(X.keys())
    yNames = list(Y.dtype.names)
    assert set(yNames) - set(xNames)  == set(['time']), (yNames, xNames)

    # Test for equality in some tables.
    for i in range(1, 10):
        a, b = Y[xNames[i]], X[xNames[i]]
        assert a.shape == b.shape, (a.shape, b.shape)
        assert (a == b).all(), (a-b)


def main( ):
    test_small( )
    test_large_system()
    print( '[INFO] All tests passed' )

if __name__ == '__main__':
    main()
