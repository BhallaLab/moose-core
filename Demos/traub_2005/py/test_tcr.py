# test_tcr.py --- 
# 
# Filename: test_tcr.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Mon Jul 16 17:08:04 2012 (+0530)
#           By: subha
#     Update #: 77
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

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')

import unittest
import uuid

import pylab
import moose

import cells

def setupClocks(dt):
    print 'Setting up clocks'
    for ii in range(10):
        moose.setClock(ii, dt)

def setupCurrentStepModel(testId, celltype, pulsearray, dt):
    """Setup a single cell simulation.

    simid - integer identifying the model

    celltype - str cell type
    
    pulsearray - an nx3 array with row[i] = (delay[i], width[i], level[i]) of current injection.
    """
    modelContainer = moose.Neutral('/test%d' % (testId))
    dataContainer = moose.Neutral('/data%d' % (testId))
    cell = cells.TCR('%s/TCR' % (modelContainer.path))
    pulsegen = moose.PulseGen('%s/pulse' % (modelContainer.path))
    pulsegen.count = len(pulsearray)
    for ii in range(len(pulsearray)):
        pulsegen.delay[ii] = pulsearray[ii][0]
        pulsegen.width[ii] = pulsearray[ii][1]
        pulsegen.level[ii] = pulsearray[ii][2]
    moose.connect(pulsegen, 'outputOut', cell.soma, 'injectMsg')
    somaVm = moose.Table('%s/vm' % (dataContainer.path))
    moose.connect(somaVm, 'requestData', cell.soma, 'get_Vm')
    pulseTable = moose.Table('%s/pulse' % (dataContainer.path))
    moose.connect(pulseTable, 'requestData', pulsegen, 'get_output')
    setupClocks(dt)
    moose.useClock(0, '%s/##[ISA=Compartment]' % (cell.path), 'init')
    moose.useClock(1, '%s/##[ISA=Compartment]' % (cell.path), 'process')
    moose.useClock(7, pulsegen.path, 'process')
    moose.useClock(8, somaVm.path, 'process')
    return {'cell': cell,
            'stimulus': pulsegen,
            'vmTable': somaVm,
            'stimTable': pulseTable
            }
    
def runsim(simtime):
    moose.reinit()
    moose.start(simtime)

pulsearray = [[1.0, 100e-3, 0.9e-9],
              [0.5, 100e-3, 0.3e-9],
              [0.5, 100e-3, 0.1e-9],
              [0.5, 100e-3, -0.1e-9],
              [0.5, 100e-3, -0.3e-9]]
simdt = 1e-6
simtime = 5e-3

class TestTCR(unittest.TestCase):
    def setUp(self):
        self.testId = uuid.uuid4().int
        params = setupCurrentStepModel(self.testId, 'TCR', pulsearray, simdt)
        print 'Starting simulation'
        runsim(simtime)
        tseries = np.arrange(0, simtime, len(params['vm'].vec))
        pylab.plot(tseries, params['vmTable'].vec * 1e3, label='Vm (mV)')
        pylab.plot(tseries, params['stimTable'].vec * 1e-12, label='Stimulus (pA)')
        pylab.show()
    def testDefault(self):
        pass

if __name__ == '__main__':
    unittest.main()



# 
# test_tcr.py ends here
