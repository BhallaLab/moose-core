#!/usr/bin/env python
# This is test for createmap and planarconnect command in Python.
import sys
import math

# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
sys.path.append('..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

context = moose.PyMooseBase.getContext()
cell = moose.Neutral("/cell")
comp = moose.Compartment("/comp")
spike = moose.SpikeGen("/comp/spike")
spike.threshold = 0.0
spike.absRefractT = 10e-3
spike.amplitude = 1
chan = moose.SynChan("/comp/xchan") # excitatory channel 
chan.Ek = 0.045
chan.tau1 = 3.0e-3
chan.tau2 = 3.0e-3
chan.gmax = 1e-9*50
context.createMap(comp.id, cell.id, "c", 2, 4)
for id in cell.children():
    print id.path()

context.planarConnect("/cell/c[]/spike","/cell/c[]/xchan", 0.5)
for comp in cell.children():
    spike = moose.SpikeGen(comp.path()+"/spike")
    for msg in spike.outMessages():
        print msg


context.planarDelay("/cell/c[]/spike",  0.005)
