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
    msg += " outfile : %s" % table.outfile 
    msg += " useStreamer: %s" % table.useStreamer 
    msg += ' Path: %s' % table.path 
    print( msg )

def sanity_test( ):
    a = moose.Table( '/t1' )
    b = moose.Table( '/t1/t1' )
    c = moose.Table( '/t1/t1/t1' )
    tables = [ a, b, c ]
    assert a.useStreamer == 0

    [ print_table( x ) for x in tables ]
    b.useStreamer = 1

    moose.reinit()
    [ print_table( x ) for x in tables ]
    assert b.useStreamer == 1
    print( '[TEST 1] passed ' )


def test( ):
    compt = moose.CubeMesh( '/compt' )
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

    tabA = moose.Table2( '/compt/a/tab' )
    tabA.useStreamer = 1

    tabB = moose.Table2( '/compt/tabB' )
    tabB.outfile = 'table2.dat'

    tabC = moose.Table2( '/compt/tabB/tabC' )

    moose.connect( tabA, 'requestOut', a, 'getConc' )
    moose.connect( tabB, 'requestOut', b, 'getConc' )
    moose.connect( tabC, 'requestOut', c, 'getConc' )

    moose.reinit( )
    [ print_table( x) for x in [tabA, tabB, tabC] ]
    moose.start( 57 )

    print( '[TEST 2] Passed' )

def main( ):
    sanity_test( )
    test( )
    print( '[INFO] All tests passed' )


if __name__ == '__main__':
    main()
