# cell_test_util.py --- 
# 
# Filename: cell_test_util.py
# Description: Utility functions for testing single cells
# Author: 
# Maintainer: 
# Created: Mon Oct 15 15:03:09 2012 (+0530)
# Version: 
# Last-Updated: Mon Oct 15 15:45:19 2012 (+0530)
#           By: subha
#     Update #: 72
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

import uuid
import unittest
import numpy as np
from matplotlib import pyplot as plt
import moose
import cells
from testutils import compare_cell_dump, setup_clocks, assign_clocks


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
    hsolve = None
    if solver == 'hsolve':
        hsolve = moose.HSolve(model_container.path+'/hsolve')
        print 'Created', hsolve.path
        hsolve.dt = simdt
        hsolve.target = cell.path
    return {'cell': cell,
            'stimulus': pulsegen,
            'presynapticVm': presyn_vm,
            'somaVm': soma_vm,
            'injectionCurrent': pulse_table,
            'hsolve': hsolve
            }


class SingleCellCurrentStepTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
    
    def setUp(self):
        self.test_id = uuid.uuid4().int
        self.test_container = moose.Neutral('test%d' % (self.test_id))
        self.model_container = moose.Neutral('%s/model' % (self.test_container.path))
        self.data_container = moose.Neutral('%s/data' % (self.test_container.path))
    
# 
# cell_test_util.py ends here
