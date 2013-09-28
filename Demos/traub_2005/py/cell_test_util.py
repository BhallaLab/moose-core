# cell_test_util.py --- 
# 
# Filename: cell_test_util.py
# Description: Utility functions for testing single cells
# Author: 
# Maintainer: 
# Created: Mon Oct 15 15:03:09 2012 (+0530)
# Version: 
# Last-Updated: Thu Jul 18 18:36:07 2013 (+0530)
#           By: subha
#     Update #: 299
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


def setup_current_step_model(model_container, 
                             data_container, 
                             celltype, 
                             pulsearray):
    """Setup a single cell simulation.

    model_container: element to hold the model

    data_container: element to hold data
    

    celltype: str - cell type
    
    pulsearray: nx3 array - with row[i] = (delay[i], width[i],
    level[i]) of current injection.

    simdt: float - simulation time step

    plotdt: float - sampling interval for plotting

    solver: str - numerical method to use, can be `hsolve` or `ee`
    """
    cell_class = eval('cells.%s' % (celltype))
    cell = cell_class('%s/%s' % (model_container.path, celltype))
    print '111111', cell.path
    pulsegen = moose.PulseGen('%s/pulse' % (model_container.path))
    print '121221211', pulsegen.path
    pulsegen.count = len(pulsearray)
    print '7777777', pulsegen.count, pulsegen.id_
    for ii in range(len(pulsearray)):
        print '999999', pulsegen.id_, pulsegen.count
        print '-', pulsegen.delay[ii]
        pulsegen.delay[ii] = pulsearray[ii][0]
        pulsegen.width[ii] = pulsearray[ii][1]
        pulsegen.level[ii] = pulsearray[ii][2]
        print '8888888', ii, pulsegen.delay[ii]
    moose.connect(pulsegen, 'outputOut', cell.soma, 'injectMsg')
    print '22222'
    presyn_vm = moose.Table('%s/presynVm' % (data_container.path))
    print '33333'
    soma_vm =  moose.Table('%s/somaVm' % (data_container.path))
    print '44444'
    moose.connect(presyn_vm, 'requestData', cell.presynaptic, 'get_Vm')
    moose.connect(soma_vm, 'requestData', cell.soma, 'get_Vm')
    pulse_table = moose.Table('%s/injectCurrent' % (data_container.path))
    moose.connect(pulse_table, 'requestData', pulsegen, 'get_output')
    print '55555'
    return {'cell': cell,
            'stimulus': pulsegen,
            'presynVm': presyn_vm,
            'somaVm': soma_vm,
            'injectionCurrent': pulse_table, }


class SingleCellCurrentStepTest(unittest.TestCase):
    """Base class for simulating a single cell with step current
    injection"""
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)        
        self.pulse_array =  [[100e-3, 100e-3, 1e-9],
                             [1e9, 0, 0]]
        self.solver = None
        self.simdt = None
        self.plotdt = None
        self.tseries = []
    
    def setUp(self):
        self.test_id = uuid.uuid4().int
        self.test_container = moose.Neutral('test%d' % (self.test_id))
        self.model_container = moose.Neutral('%s/model' % (self.test_container.path))
        self.data_container = moose.Neutral('%s/data' % (self.test_container.path))    
        params = setup_current_step_model(
            self.model_container, 
            self.data_container, 
            self.celltype, 
            self.pulse_array)
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

    def schedule(self, simdt, plotdt, solver):
        config.logger.info('Scheduling: simdt=%g, plotdt=%g, solver=%s' % (simdt, plotdt, solver))
        self.simdt = simdt
        self.plotdt = plotdt
        self.solver = solver
        if self.solver == 'hsolve':
            self.hsolve = moose.HSolve('%s/solver' % (self.cell.path))
            self.hsolve.dt = simdt
            self.hsolve.target = self.cell.path
        mutils.setDefaultDt(elecdt=simdt, plotdt2=plotdt)
        mutils.assignDefaultTicks(modelRoot=self.model_container.path, 
                                 dataRoot=self.data_container.path, 
                                 solver=self.solver)        

    def runsim(self, simtime, stepsize=0.1, pulsearray=None):
        """Run the simulation for `simtime`. Save the data at the
        end."""
        config.logger.info('running: simtime=%g, stepsize=%g, pulsearray=%s' % (simtime, stepsize, str(pulsearray)))
        self.simtime = simtime
        if pulsearray is not None:            
            self.tweak_stimulus(pulsearray)
        for ii in range(self.pulsegen.count):
            config.logger.info('pulse[%d]: delay=%g, width=%g, level=%g' % (ii, self.pulsegen.delay[ii], self.pulsegen.width[ii], self.pulsegen.level[ii]))
        config.logger.info('Start reinit')
        self.schedule(self.simdt, self.plotdt, self.solver)
        moose.reinit()
        config.logger.info('Finished reinit')
        ts = datetime.now()
        mutils.stepRun(simtime, simtime/10.0, verbose=True, logger=config.logger)
        # The sleep is required to get all threads to end 
        while moose.isRunning():
            time.sleep(0.1)
        te = datetime.now()
        td = te - ts
        config.logger.info('Simulation time of %g s at simdt=%g with solver %s: %g s' % \
            (simtime, self.simdt, self.solver, 
             td.seconds + td.microseconds * 1e-6))

    def savedata(self):
        # Now save the data
        for table_id in self.data_container.children:
            ts = np.linspace(0, self.simtime, len(table_id[0].vec))
            data = np.vstack((ts, table_id[0].vec))
            fname = os.path.join(config.data_dir, 
                                 '%s_%s_%s_%s.dat' % (self.celltype, 
                                                      table_id[0].name,
                                                      self.solver, 
                                                      config.filename_suffix))
            np.savetxt(fname, np.transpose(data))
            config.logger.info('Saved %s in %s' % (table_id[0].name, fname))
        
    def plot_vm(self):
        """Plot Vm for presynaptic compartment and soma - along with
        the same in NEURON simulation if possible."""
        pylab.subplot(211)
        pylab.title('Soma Vm')
        self.tseries = np.linspace(0, self.simtime, len(self.somaVmTab.vec))
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
