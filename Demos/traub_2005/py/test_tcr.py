# test_tcr.py --- 
# 
# Filename: test_tcr.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Mon Aug 27 16:59:35 2012 (+0530)
#           By: subha
#     Update #: 203
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 

# Code:

from datetime import datetime
import time
import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')

import unittest
import uuid

import numpy as np
import pylab
import moose
from testutils import compare_cell_dump
import cells

def setupClocks(dt):
    print 'Setting up clocks'
    for ii in range(10):
        moose.setClock(ii, dt)

def setupCurrentStepModel(testId, celltype, pulsearray, dt, solver=None):
    """Setup a single cell simulation.

    simid - integer identifying the model

    celltype - str cell type
    
    pulsearray - an nx3 array with row[i] = (delay[i], width[i], level[i]) of current injection.
    """
    modelContainer = moose.Neutral('/test%d' % (testId))
    dataContainer = moose.Neutral('/data%d' % (testId))
    cell = cells.TCR('%s/TCR' % (modelContainer.path)) # moose.copy(cells.TCR.prototype, modelContainer.path)#
    if solver == 'hsolve':
        hsolve = moose.HSolve(cell.path + '/solve')
        hsolve.dt = dt
        hsolve.seed = cell.soma
        hsolve.target = cell.path
    pulsegen = moose.PulseGen('%s/pulse' % (modelContainer.path))
    pulsegen.count = len(pulsearray)
    for ii in range(len(pulsearray)):
        pulsegen.delay[ii] = pulsearray[ii][0]
        pulsegen.width[ii] = pulsearray[ii][1]
        pulsegen.level[ii] = pulsearray[ii][2]
    moose.connect(pulsegen, 'outputOut', cell.soma, 'injectMsg')
    presynVm = moose.Table('%s/presynVm' % (dataContainer.path))
    somaVm =  moose.Table('%s/somaVm' % (dataContainer.path))
    moose.connect(presynVm, 'requestData', cell.presynaptic, 'get_Vm')
    moose.connect(somaVm, 'requestData', cell.soma, 'get_Vm')
    pulseTable = moose.Table('%s/pulse' % (dataContainer.path))
    moose.connect(pulseTable, 'requestData', pulsegen, 'get_output')
    setupClocks(dt)
    moose.useClock(0, '%s/##[ISA=Compartment]' % (cell.path), 'init')
    moose.useClock(1, '%s/##[ISA=Compartment]' % (cell.path), 'process')
    moose.useClock(2, '%s/##[ISA=HHChannel]' % (cell.path), 'process')
    moose.useClock(3, '%s/##[ISA=CaConc]' % (cell.path), 'process')
    moose.useClock(7, pulsegen.path, 'process')
    moose.useClock(8, '%s/##' % (dataContainer.path), 'process')
    return {'cell': cell,
            'stimulus': pulsegen,
            'presynapticVm': presynVm,
            'somaVm': somaVm,
            'stimTable': pulseTable
            }
    
def runsim(simtime, steptime=0.1):
    moose.reinit()
    clock = moose.Clock('/clock')
    # while clock.currentTime < simtime - steptime:
    #     moose.start(steptime)
    #     print 't =', clock.currentTime
    #     time.sleep(0.005)
    moose.start(simtime - clock.currentTime)

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]

pulsearray = [[100e-3, 100e-3, 1e-9],
              [1e9, 0, 0]]

simdt = 0.25e-4
simtime = 1.0

class TestTCR(unittest.TestCase):
    def setUp(self):
        self.testId = uuid.uuid4().int
        params = setupCurrentStepModel(self.testId, 'TCR', pulsearray, simdt)
        dump_file = 'data/TCR.csv'
        params['cell'].dump_cell(dump_file)
        compare_cell_dump(dump_file, '../nrn/'+dump_file)
        print 'Starting simulation'
        start = datetime.now()
        runsim(simtime)
        end = datetime.now()
        delta = end - start
        print 'Simulation time:', delta.seconds + delta.microseconds * 1e-6
        tseries = np.linspace(0, simtime, len(params['somaVm'].vec))
        np.savetxt('data/TCR_soma_Vm.dat', np.transpose(np.vstack((tseries, params['somaVm'].vec))))
        np.savetxt('data/TCR_presynaptic_Vm.dat', np.transpose(np.vstack((tseries, params['presynapticVm'].vec))))
        print 'Soma Em:', params['cell'].soma.Em
        pylab.subplot(211)
        pylab.title('Soma Vm')
        pylab.plot(tseries*1e3, params['somaVm'].vec * 1e3, label='Vm (mV) - moose')
        pylab.plot(tseries*1e3, params['stimTable'].vec * 1e9, label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/TCR_soma_Vm.dat')
            pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.subplot(212)
        pylab.title('Presynaptic Vm')
        pylab.plot(tseries*1e3, params['presynapticVm'].vec * 1e3, label='Vm (mV) - moose')
        pylab.plot(tseries*1e3, params['stimTable'].vec * 1e9, label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/TCR_presynaptic_Vm.dat')
            pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.show()

    def testDefault(self):
        pass

if __name__ == '__main__':
    unittest.main()



# 
# test_tcr.py ends here
