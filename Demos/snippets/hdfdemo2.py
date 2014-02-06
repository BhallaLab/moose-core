#!/usr/bin/env python
# Author: Subhasis Ray

"""Demonstrates the use of HDF5DataWriter class to save simulated
data in HDF5 file under different groups reflecting the organization
of the table objects.

In this example a passive neuronal compartment `c` is created and its
membrane voltage Vm and membrane current Im are recorded in two tables
`t` and `t1` respectively. The current time is recorded under
`t2`. The tables are under different element and all the tables are
in turn connected to a single HDF5DataWriter object, which saves the
table contents into an HDF5 file `hdfdemo2.h5`.

The `clearOut` message of the HDF5 object is connected to each table's
`clearVec` destination, so that after the data has been recorded by
the HDF5 object, the tables are emptied.

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
d1 = moose.Neutral('/data1')
t = moose.Table('/data1/Vm')
d2 = moose.Neutral('/data2')
t1 = moose.Table('/data2/Im')
t2 = moose.Table('/data2/currentTime')
moose.connect(t, 'requestOut', c, 'getVm')
moose.connect(t1, 'requestOut', c, 'getIm')
# This is a bad example : the currentTime recorded gets messed up at
# the ticks of the slowest clock.
moose.connect(t2, 'requestOut', moose.element('/clock'), 'getCurrentTime')
h = moose.HDF5DataWriter('h')
h.mode = 2 # Truncate existing file
moose.connect(h, 'requestOut', t, 'getVec')
moose.connect(h, 'requestOut', t1, 'getVec')
moose.connect(h, 'requestOut', t2, 'getVec')
# Ensure the vectors are cleared after HDF5DataWriter is done with
# saving at each time step.
moose.connect(h, 'clearOut', t, 'clearVec')
moose.connect(h, 'clearOut', t1, 'clearVec')
moose.connect(h, 'clearOut', t2, 'clearVec')
h.filename = 'hdfdemo2.h5'
moose.setClock(0, 0.1)
moose.setClock(1, 0.1)
moose.setClock(2, 10)
moose.useClock(0, '/c', 'init')
moose.useClock(1, '/##[TYPE!=HDF5DataWriter]', 'process')
moose.useClock(2, '/##[TYPE=HDF5DataWriter]', 'process')
moose.reinit()
c.inject = 0.1
moose.start(30.0)
print 'Finished simulation. Data was saved in', h.filename

