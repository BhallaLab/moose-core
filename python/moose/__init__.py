"""
How to use the documentation
----------------------------

MOOSE documentation is split into Python documentation and builtin
documentation. The functions and classes that are only part of the
Python interface can be viewed via Python's builtin ``help``
function::

>>> help(moose.connect)

The documentation built into main C++ code of MOOSE can be accessed
via the module function ``doc``::

>>> moose.doc('Neutral')

To get documentation about a particular field::

>>> moose.doc('Neutral.childMsg')

Builtin functions and classes in moose module (Python only)
-----------------------------------------------------------

"""
from .moose import *
# import genesis

import os
cmake_file = os.path.dirname(moose.__file__)+'/../../CMakeLists.txt'

if os.path.isfile(cmake_file):
    for line in open(cmake_file):
        if 'MOOSE_VERSION' in line:
            __version__ = line.split('"')[1]
else:
    print("Cannot set Moose version")

    

