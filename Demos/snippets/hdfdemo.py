#!/usr/bin/env python
# Author: Subhasis Ray

"""
Demonstrates the use of HDF5DataWriter class to save simulated data in
HDF5 file.

In this example a passive neuronal compartment `c` is created and its
membrane voltage Vm and membrane current Im are recorded in two tables
`t` and `t1` respectively. Both the tables are in turn connected to a
single HDF5DataWriter object, which saves the table contents into an
HDF5 file `output.h5`. 

The `mode` of the HDF5DataWriter is set to `2` which means if a file
of the same name exists, it will be overwritten.
"""
import sys
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import numpy
import moose

c = moose.Compartment('c')
t = moose.Table('t')
t1 = moose.Table('t1')
moose.connect(t, 'requestOut', c, 'getVm')
moose.connect(t1, 'requestOut', c, 'getIm')
h = moose.HDF5DataWriter('h')
h.mode = 2 # Truncate existing file
moose.connect(h, 'requestOut', t, 'getVec')
moose.connect(h, 'requestOut', t1, 'getVec')
h.filename = 'output.h5'
h.chunkSize = 10000
h.compressor = 'zlib'
h.compression = 7
# We allow simple attributes of type string, double and long on the
# root node. This allows for file-level metadata/annotation.
h.sattr['note'] = 'This is a test.'
h.fattr['a float attribute'] = 3.141592
h.iattr['an int attribute'] = 86400
moose.setClock(0, 1e-5)
moose.setClock(1, 1e-5)
moose.setClock(2, 1.0)
moose.useClock(0, '/c', 'init')
moose.useClock(1, '/##[TYPE!=HDF5DataWriter]', 'process')
moose.useClock(2, '/##[TYPE=HDF5DataWriter]', 'process')
moose.reinit()
c.inject = 0.1
moose.start(30.0)
h.close()
print 'Finished simulation. Data was saved in', h.filename
# print numpy.array(t.vector)
# moose.start(0.5)
# print numpy.array(t.vector)
