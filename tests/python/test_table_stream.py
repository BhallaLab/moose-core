"""test_table_stream.py: 

    Test moose.TableStream.

    This class takes over moose.Table and make sure that their vector is written
    to stream. Once a chunk of data is written, it is removed from vector.

    This way, moose never runs out of memory during simulation.

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

print( '[INFO] Using moose from %s' % moose.__file__ )

a = moose.TableStream( '/a' )
assert( a.streamname == 'stdout' ), 'default is stdout, got %s' % a.streamname
