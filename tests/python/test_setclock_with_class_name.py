# -*- coding: utf-8 -*-
"""test_setclock_with_class_name.py:

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import moose
print( "[INFO ] Using moose form %s, %s" % (moose.__file__, moose.version()) )

def main( ):
    moose.setClock( 'Table2' , 0.1 )
    moose.setClock( 'Ksolve', 0.01 )

if __name__ == '__main__':
    main()


