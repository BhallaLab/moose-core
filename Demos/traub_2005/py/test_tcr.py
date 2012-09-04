# test_tcr.py --- 
# 
# Filename: test_tcr.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Tue Sep  4 09:55:40 2012 (+0530)
#           By: subha
#     Update #: 424
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
from moose import utils

INITCLOCK = 0
ELECCLOCK = 1
CHANCLOCK = 2
POOLCLOCK = 3
LOOKUPCLOCK = 6
STIMCLOCK = 7
PLOTCLOCK = 8

SIMDT = 5e-6
PLOTDT = 0.25e-3
SIMTIME = 1.0

def setupClocks(simdt, plotdt):
    print 'Setting up clocks: simdt', simdt, 'plotdt', plotdt
    moose.setClock(INITCLOCK, simdt)
    moose.setClock(ELECCLOCK, simdt)
    moose.setClock(CHANCLOCK, simdt)
    moose.setClock(POOLCLOCK, simdt)
    moose.setClock(LOOKUPCLOCK, simdt)
    moose.setClock(STIMCLOCK, simdt)
    moose.setClock(PLOTCLOCK, plotdt)
    
def setupCurrentStepModel(testId, celltype, pulsearray, simdt, plotdt, solver='euler'):
    """Setup a single cell simulation.

    simid - integer identifying the model

    celltype - str cell type
    
    pulsearray - an nx3 array with row[i] = (delay[i], width[i], level[i]) of current injection.
    """
    modelContainer = moose.Neutral('/test%d' % (testId))
    dataContainer = moose.Neutral('/data%d' % (testId))
    cell_list = []
    for ii in range(1):
        cell = cells.TCR('%s/TCR_%d' % (modelContainer.path, ii)) 
        cell_list.append(cell)
    pulsegen = moose.PulseGen('%s/pulse' % (modelContainer.path))
    pulsegen.count = len(pulsearray)
    for ii in range(len(pulsearray)):
        pulsegen.delay[ii] = pulsearray[ii][0]
        pulsegen.width[ii] = pulsearray[ii][1]
        pulsegen.level[ii] = pulsearray[ii][2]
    for ii, cell in enumerate(cell_list):
        moose.connect(pulsegen, 'outputOut', cell.soma, 'injectMsg')
        presynVm = moose.Table('%s/presynVm_%d' % (dataContainer.path, ii))
        somaVm =  moose.Table('%s/somaVm_%d' % (dataContainer.path, ii))
        moose.connect(presynVm, 'requestData', cell.presynaptic, 'get_Vm')
        moose.connect(somaVm, 'requestData', cell.soma, 'get_Vm')
    pulseTable = moose.Table('%s/pulse' % (dataContainer.path))
    moose.connect(pulseTable, 'requestData', pulsegen, 'get_output')
    setupClocks(simdt, plotdt)
    moose.useClock(STIMCLOCK, modelContainer.path+'/##[TYPE=PulseGen]', 'process')
    moose.useClock(PLOTCLOCK, dataContainer.path+'/##[TYPE=Table]', 'process')
    if solver == 'hsolve':
        for ii in range(len(cell_list)):
            hsolve = moose.HSolve(modelContainer.path + '/solve_%d' % (ii))
            print 'Created', hsolve.path
            hsolve.dt = simdt
            hsolve.target = cell_list[ii].path
        moose.useClock(INITCLOCK, modelContainer.path+'/##[TYPE=HSolve]', 'process')
    else:
        moose.useClock(INITCLOCK, modelContainer.path+'/##[TYPE=Compartment]', 'init')
        moose.useClock(ELECCLOCK, modelContainer.path+'/##[TYPE=Compartment]', 'process')
        moose.useClock(CHANCLOCK, modelContainer.path+'/##[TYPE=HHChannel]', 'process')
        moose.useClock(POOLCLOCK, modelContainer.path+'/##[TYPE=CaConc]', 'process')

    return {'cell': cell,
            'stimulus': pulsegen,
            'presynapticVm': presynVm,
            'somaVm': somaVm,
            'stimTable': pulseTable,
            'model': modelContainer,
            'data': dataContainer
            }
    
def runsim(simtime, steptime=0.1):
    clock = moose.Clock('/clock')
    print 'Starting simulation for', simtime
    while clock.currentTime < simtime - steptime:
        moose.start(steptime)
        print 'Simulated till', clock.currentTime, 's'
    remaining = simtime % steptime
    moose.start(remaining)
    print 'Finished simulation'

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]

pulsearray = [[100e-3, 100e-3, 1e-9],
              [1e9, 0, 0]]


class TestTCR(unittest.TestCase):
    def setUp(self):
        self.testId = uuid.uuid4().int
        self.solver = 'hsolve'
        params = setupCurrentStepModel(self.testId, 'TCR', pulsearray, SIMDT, PLOTDT, solver=self.solver)
        cell = params['cell']
        moose.le('/library')
        kahp_slower = moose.element(cell.soma.path + '/KAHP_SLOWER')
        capool = moose.element(cell.soma.path + '/CaPool')
        # print '************* Messages for KAHP_SLOWER ******************'
        # for msg in kahp_slower.msgIn:
        #     print 'E1:', msg.e1
        #     print '\nSrc fields on E1:\n', msg.srcFieldsOnE1
        #     print '\nDest fields on E1:\n', msg.destFieldsOnE1
        #     print 'E2:', msg.e2
        #     print '\nSrc fields on E2:\n', msg.srcFieldsOnE2
        #     print '\nDest fields on E2:\n', msg.destFieldsOnE2
        # print '************* Messages for CaPool ******************'
        # for msg in capool.msgOut:
        #     print 'E1:', msg.e1
        #     print 'Src fields on E1:\n', msg.srcFieldsOnE1
        #     print 'Dest fields on E1:\n', msg.destFieldsOnE1
        #     print 'E2:', msg.e2
        #     print '\nSrc fields on E2:\n', msg.srcFieldsOnE2
        #     print '\nDest fields on E2:\n', msg.destFieldsOnE2
        # print 'Finished model setup'
        # self.dump_file = 'data/TCR.csv'
        # params['cell'].dump_cell(self.dump_file)
        moose.reinit()
        start = datetime.now()
        runsim(SIMTIME)
        end = datetime.now()
        delta = end - start
        print 'Simulation time:', delta.seconds + delta.microseconds * 1e-6
            
        tseries = np.linspace(0, SIMTIME, len(params['somaVm'].vec))
        for table_id in params['data'].children:
            np.savetxt('data/%s_Vm_%s.dat' % (table_id[0].name, self.solver), np.transpose(np.vstack((tseries, table_id[0].vec))))
        # np.savetxt('data/TCR_soma_Vm_%s.dat' % (self.solver), np.transpose(np.vstack((tseries, params['somaVm'].vec))))
        # np.savetxt('data/TCR_presynaptic_Vm_%s.dat' % (self.solver), np.transpose(np.vstack((tseries, params['presynapticVm'].vec))))
        # pylab.subplot(211)
        # pylab.title('Soma Vm')
        # pylab.plot(tseries*1e3, params['somaVm'].vec * 1e3, label='Vm (mV) - moose')
        # pylab.plot(tseries*1e3, params['stimTable'].vec * 1e9, label='Stimulus (nA)')
        # try:
        #     nrn_data = np.loadtxt('../nrn/data/TCR_soma_Vm.dat')
        #     pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        # except IOError:
        #     print 'No neuron data found.'
        # pylab.legend()
        # pylab.subplot(212)
        # pylab.title('Presynaptic Vm')
        # pylab.plot(tseries*1e3, params['presynapticVm'].vec * 1e3, label='Vm (mV) - moose')
        # pylab.plot(tseries*1e3, params['stimTable'].vec * 1e9, label='Stimulus (nA)')
        # try:
        #     nrn_data = np.loadtxt('../nrn/data/TCR_presynaptic_Vm.dat')
        #     pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        # except IOError:
        #     print 'No neuron data found.'
        # pylab.legend()
        # pylab.show()

    def testChannelDensities(self):
        pass
        # equal = compare_cell_dump(self.dump_file, '../nrn/'+self.dump_file)
        # self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()



# 
# test_tcr.py ends here
