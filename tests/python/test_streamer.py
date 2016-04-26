"""test_streamer.py: 

Test script for Streamer class.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2016, Dilawar Singh"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import moose
print( '[INFO] Using moose form %s' % moose.__file__ )

def main( ):
    a = moose.Table( '/t1' )
    b = moose.Table( '/t1/t1' )
    c = moose.Table( '/t1/t1/t1' )

    st = moose.Streamer( '/s' )
    assert st.streamname == 'stdout', 'Expecting stdout, got %s' % st.streamname

    st.streamname = 'a.txt'
    assert st.streamname == 'a.txt'

    st.addTable( a )
    assert( st.numTables == 1 )
    st.addTable( b )
    assert( st.numTables == 2 )
    st.addTable( c )
    assert( st.numTables == 3 )
    st.addTable( c )
    assert( st.numTables == 3 )
    st.addTable( c )
    assert( st.numTables == 3 )

    st.removeTable( c )
    assert( st.numTables == 2 )
    st.removeTable( c )
    assert( st.numTables == 2 )
    st.removeTable( a )
    assert( st.numTables == 1 )
    st.removeTable( b )
    assert( st.numTables == 0 )
    st.removeTable( b )
    assert( st.numTables == 0 )


if __name__ == '__main__':
    main()
