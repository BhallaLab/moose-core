# test_tcr.py --- 
# 
# Filename: test_tcr.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Mon Oct 15 16:18:01 2012 (+0530)
#           By: subha
#     Update #: 465
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
import testutils
from testutils import compare_cell_dump, setup_clocks, assign_clocks, step_run
from cell_test_util import setup_current_step_model, SingleCellCurrentStepTest
import cells
from moose import utils

simdt = 5e-6
plotdt = 0.25e-3
simtime = 1.0
    

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]

pulsearray = [[100e-3, 100e-3, 1e-9],
              [1e9, 0, 0]]


class TestTCR(SingleCellCurrentStepTest):
    def setUp(self):
        SingleCellCurrentStepTest.setUp(self)
        self.solver = 'hsolve'
        params = setup_current_step_model(
            self.model_container, 
            self.data_container, 
            'TCR', 
            pulsearray, 
            testutils.SIMDT, 
            testutils.PLOTDT,
            solver=self.solver)
        self.cell = params['cell']
        testutils.setup_clocks(testutils.SIMDT, testutils.PLOTDT)
        testutils.assign_clocks(self.model_container, 
                                self.data_container, 
                                solver=self.solver)
        moose.reinit()
        start = datetime.now()
        step_run(simtime, 0.1)
        end = datetime.now()
        delta = end - start
        print 'Simulation time:', delta.seconds \
            + delta.microseconds * 1e-6            
        tseries = np.linspace(0, simtime, len(params['somaVm'].vec))
        for table_id in self.data_container.children:
            data = np.vstack((tseries, table_id[0].vec))
            np.savetxt('data/TCR_%s_%s.dat' % (table_id[0].name, self.solver), 
                       np.transpose(data))
        pylab.subplot(211)
        pylab.title('Soma Vm')
        pylab.plot(tseries*1e3, params['somaVm'].vec * 1e3, label='Vm (mV) - moose')
        pylab.plot(tseries*1e3, params['injectionCurrent'].vec * 1e9, label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/TCR_soma_Vm.dat')
            pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.subplot(212)
        pylab.title('Presynaptic Vm')
        pylab.plot(tseries*1e3, params['presynapticVm'].vec * 1e3, label='Vm (mV) - moose')
        pylab.plot(tseries*1e3, params['injectionCurrent'].vec * 1e9, label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/TCR_presynaptic_Vm.dat')
            pylab.plot(nrn_data[:,0], nrn_data[:,1], label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.show()

    def testChannelDensities(self):
        pass
        # equal = compare_cell_dump(self.dump_file, '../nrn/'+self.dump_file)
        # self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()



# 
# test_tcr.py ends here
