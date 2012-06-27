# import moose.utils as mu
# mu.printtree('/')
import numpy
import moose
c = moose.Compartment('c')
t = moose.Table('t')
moose.connect(t, 'requestData', c, 'get_Vm')
h = moose.HDF5DataWriter('h')
moose.connect(h, 'requestData', t, 'get_vec')
h.filename = 'output.h5'
moose.setClock(0, 0.1)
moose.setClock(1, 0.1)
moose.useClock(0, '/c', 'init')
moose.useClock(1, '/##', 'process')
moose.reinit()
c.inject = 0.1
moose.start(1.0)
# print numpy.array(t.vec)
# moose.start(0.5)
# print numpy.array(t.vec)
