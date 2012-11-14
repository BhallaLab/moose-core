# import moose.utils as mu
# mu.printtree('/')
import sys
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import numpy
import moose
c = moose.Compartment('c')
t = moose.Table('t')
t1 = moose.Table('t1')
moose.connect(t, 'requestData', c, 'get_Vm')
moose.connect(t1, 'requestData', c, 'get_Im')
h = moose.HDF5DataWriter('h')
moose.connect(h, 'requestData', t, 'get_vec')
moose.connect(h, 'requestData', t1, 'get_vec')
h.filename = 'output.h5'
moose.setClock(0, 0.1)
moose.setClock(1, 0.1)
moose.setClock(2, 10)
moose.useClock(0, '/c', 'init')
moose.useClock(1, '/##[TYPE!=HDF5DataWriter]', 'process')
moose.useClock(2, '/##[TYPE=HDF5DataWriter]', 'process')
moose.reinit()
c.inject = 0.1
moose.start(30.0)
print 'Here'
# print numpy.array(t.vec)
# moose.start(0.5)
# print numpy.array(t.vec)
