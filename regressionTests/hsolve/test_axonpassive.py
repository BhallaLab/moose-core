# test_axonpassive.py --- 
# 
# Filename: test_axonpassive.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Jul 11 00:03:30 2012 (+0530)
# Version: 
# Last-Updated: Wed Jul 11 01:46:02 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 59
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

# Code:

import sys
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'

import numpy as np
from matplotlib import pyplot as plt

import moose

simdt = 1e-6
iodt = 1e-6
simlength = 0.05
inject = 5e-10
erest_act = -0.065

def load_axon():
    model = moose.loadModel('axon_passive.p', '/axon')
    for x in model[0].children:
        print x.path, x.class_
    pulsegen = moose.PulseGen('/pulsegen')
    pulsegen.delay[0] = simlength * 0.2
    pulsegen.width[0] = simlength
    pulsegen.level[0] = inject
    moose.connect(pulsegen, 'outputOut', moose.element('/axon/soma'), 'injectMsg')
    data = moose.Neutral('/data')
    tab = moose.Table('%s/Vm100' % (data.path))
    moose.connect(tab, 'requestData', moose.ObjId('/axon/c100'), 'get_Vm')
    solver = moose.HSolve('/hsolve')
    solver.dt = simdt
    solver.target = model.path
    return {'model': model,
            'Vm': tab,
            'soma': moose.element('/axon/soma'),
            'pulse': pulsegen,
            'solver': solver}

def run(params):
    moose.setClock(0, simdt)
    moose.setClock(1, simdt)
    moose.setClock(2, simdt)
    moose.setClock(3, simdt)
    moose.useClock(0, params['solver'].path, 'process')
    moose.useClock(1, params['model'].path + '/##[ISA=Compartment]', 'init')
    moose.useClock(2, params['model'].path + '/##[ISA=Compartment]', 'process')
    moose.useClock(3, params['Vm'].path, 'process')
    moose.reinit()
    moose.start(simlength)
    np.savetxt('moose_axon_passive_Vm.dat', params['Vm'].vec)
    plt.plot(np.linspace(0, simlength, len(params['Vm'].vec)), params['Vm'].vec)
    plt.show()
    
if __name__ == '__main__':
    params = load_axon()
    run(params)
# 
# test_axonpassive.py ends here
