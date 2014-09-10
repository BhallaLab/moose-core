"""==================================      
PyMOOSE : moose as a python module
==================================

This document describes classes and functions specific to the MOOSE
Python module. This is an API reference.

* If you are looking for basic tutorials for getting started with 
  moose, then check :doc:`moose_quickstart`.

* If you want recipes for particular tasks, check out
  :doc:`moose_cookbook`.

* If you want the reference for specific moose classes, then go to
  :doc:`moose_classes`.

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


Documentation
=============

"""

__version__ = '$Revision: 4454$'
# $Source$

from .moose import *
