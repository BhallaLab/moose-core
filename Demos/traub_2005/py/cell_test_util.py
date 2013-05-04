# cell_test_util.py --- 
# 
# Filename: cell_test_util.py
# Description: Utility functions for testing single cells
# Author: 
# Maintainer: 
# Created: Mon Oct 15 15:03:09 2012 (+0530)
# Version: 
# Last-Updated: Fri May  3 11:53:10 2013 (+0530)
#           By: subha
#     Update #: 235
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
import uuid
import unittest
import numpy as np
from matplotlib import pyplot as plt
import pylab
import moose
from moose import utils as mutils
import config
import cells
import testutils
from testutils import compare_cell_dump, setup_clocks, assign_clocks, step_run


def setup_current_step_model(model_container, data_container, 
                                        celltype, 
                                        pulsearray, 
                                        simdt, plotdt, solver='euler'):
    """Setup a single cell simulation.

    testId: integer - identifying the model

    celltype: str - cell type
    
    pulsearray: nx3 array - with row[i] = (delay[i], width[i],
    level[i]) of current injection.

    simdt: float - simulation time step

    plotdt: float - sampling interval for plotting

    solver: str - numerical method to use, can be `hsolve` or `ee`
    """
    cell_class = eval('cells.%s' % (celltype))
    cell = cell_class('%s/%s' % (model_container.path, celltype))
    pulsegen = moose.PulseGen('%s/pulse' % (model_container.path))
    pulsegen.count = len(pulsearray)
    for ii in range(len(pulsearray)):
        pulsegen.delay[ii] = pulsearray[ii][0]
        pulsegen.width[ii] = pulsearray[ii][1]
        pulsegen.level[ii] = pulsearray[ii][2]
    moose.connect(pulsegen, 'outputOut', cell.soma, 'injectMsg')
    presyn_vm = moose.Table('%s/presynVm' % (data_container.path))
    soma_vm =  moose.Table('%s/somaVm' % (data_container.path))
    moose.connect(presyn_vm, 'requestData', cell.presynaptic, 'get_Vm')
    moose.connect(soma_vm, 'requestData', cell.soma, 'get_Vm')
    pulse_table = moose.Table('%s/injectCurrent' % (data_container.path))
    moose.connect(pulse_table, 'requestData', pulsegen, 'get_output')
    return {'cell': cell,
            'stimulus': pulsegen,
            'presynVm': presyn_vm,
            'somaVm': soma_vm,
            'injectionCurrent': pulse_table,
            # 'hsolve': hsolve
            }


class SingleCellCurrentStepTest(unittest.TestCase):
    """Base class for simulating a single cell with step current
    injection"""
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)        
        self.pulse_array =  [[100e-3, 100e-3, 1e-9],
                             [1e9, 0, 0]]
        self.solver = 'hsolve'
        self.simdt = testutils.SIMDT
        self.plotdt = testutils.PLOTDT
        self.tseries = [0.0]
    
    def setUp(self):
        self.test_id = uuid.uuid4().int
        self.test_container = moose.Neutral('test%d' % (self.test_id))
        self.model_container = moose.Neutral('%s/model' % (self.test_container.path))
        self.data_container = moose.Neutral('%s/data' % (self.test_container.path))    
        params = setup_current_step_model(
            self.model_container, 
            self.data_container, 
            self.celltype, 
            self.pulse_array, 
            self.simdt, 
            self.plotdt,
            solver=self.solver)
        self.cell = params['cell']       
        for ch in moose.wildcardFind(self.cell.soma.path + '/##[ISA=ChanBase]'):
            config.logger.debug('%s Ek = %g' % (ch.path, ch[0].Ek))
        for ch in moose.wildcardFind(self.cell.soma.path + '/##[ISA=CaConc]'):
            config.logger.debug('%s tau = %g' % (ch.path, ch[0].tau))
                                
        self.somaVmTab = params['somaVm']
        self.presynVmTab = params['presynVm']
        self.injectionTab = params['injectionCurrent']
        self.pulsegen = params['stimulus']
        # setup_clocks(self.simdt, self.plotdt)
        # assign_clocks(self.model_container, self.data_container, self.solver)        

    def tweak_stimulus(self, pulsearray):
        """Update the pulsegen for this model with new (delay, width,
        level) values specified in `pulsearray` list."""        
        for ii in range(len(pulsearray)):
            self.pulsegen.delay[ii] = pulsearray[ii][0]
            self.pulsegen.width[ii] = pulsearray[ii][1]
            self.pulsegen.level[ii] = pulsearray[ii][2]

    def runsim(self, simtime, stepsize=0.1, pulsearray=None):
        """Run the simulation for `simtime`. Save the data at the
        end."""
        mutils.resetSim([self.model_container.path, self.data_container.path], self.simdt, self.plotdt, simmethod=self.solver)
        if pulsearray is not None:            
            self.tweak_stimulus(pulsearray)
        moose.reinit()
        start = datetime.now()
        step_run(simtime, stepsize)
        end = datetime.now()
        # The sleep is required to get all threads to end 
        while moose.isRunning():
            time.sleep(0.1)
        delta = end - start
        config.logger.info('Simulation time with solver %s: %g s' % \
            (self.solver, 
             delta.seconds + delta.microseconds * 1e-6))
        self.tseries = np.arange(0, simtime+self.plotdt, self.plotdt)
        # Now save the data
        for table_id in self.data_container.children:
            try:
                data = np.vstack((self.tseries, table_id[0].vec))
            except ValueError as e:
                self.tseries = np.linspace(0, simtime, len(table_id[0].vec))
                data = np.vstack((self.tseries, table_id[0].vec))
            fname = 'data/%s_%s_%s.dat' % (self.celltype, 
                                           table_id[0].name,
                                           self.solver)
            np.savetxt(fname, np.transpose(data))
            print 'Saved', table_id[0].name, 'in', fname
        
    def plot_vm(self):
        """Plot Vm for presynaptic compartment and soma - along with
        the same in NEURON simulation if possible."""
        pylab.subplot(211)
        pylab.title('Soma Vm')
        pylab.plot(self.tseries*1e3, self.somaVmTab.vec * 1e3,
                   label='Vm (mV) - moose')
        pylab.plot(self.tseries*1e3, self.injectionTab.vec * 1e9,
                   label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/%s_soma_Vm.dat' % \
                                      (self.celltype))
            nrn_indices = np.nonzero(nrn_data[:, 0] <= self.tseries[-1]*1e3)[0]                        
            pylab.plot(nrn_data[nrn_indices,0], nrn_data[nrn_indices,1], 
                       label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.subplot(212)
        pylab.title('Presynaptic Vm')
        pylab.plot(self.tseries*1e3, self.presynVmTab.vec * 1e3,
                   label='Vm (mV) - moose')
        pylab.plot(self.tseries*1e3, self.injectionTab.vec * 1e9,
                   label='Stimulus (nA)')
        try:
            nrn_data = np.loadtxt('../nrn/data/%s_presynaptic_Vm.dat' % \
                                      (self.celltype))
            nrn_indices = np.nonzero(nrn_data[:, 0] <= self.tseries[-1]*1e3)[0]
            pylab.plot(nrn_data[nrn_indices,0], nrn_data[nrn_indices,1], 
                       label='Vm (mV) - neuron')
        except IOError:
            print 'No neuron data found.'
        pylab.legend()
        pylab.show()
        
    
# 
# cell_test_util.py ends here
