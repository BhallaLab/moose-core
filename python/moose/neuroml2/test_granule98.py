# test_granule98.py --- 
# 
# Filename: test_granule98.py
# Description: 
# Author: Subhasis Ray
# Created: Mon Apr  8 21:41:22 2024 (+0530)
# Last-Updated: Wed Jul 17 15:51:22 2024 (+0530)
#           By: Subhasis Ray
# 

# Code:
"""Test code for the Granule cell model

"""
import os
import sys
import numpy as np
# import unittest
import logging


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(level=LOGLEVEL)


import moose
from moose.neuroml2.reader import NML2Reader


def run(modeldir, nogui=True):
    reader = NML2Reader()
    filename = os.path.join(modeldir, 'GranuleCell.net.nml')
    reader.read(filename)
    soma = reader.getComp(reader.doc.networks[0].populations[0].id, 0, 0)
    data = moose.Neutral('/data')
    pg = reader.getInput('Gran_10pA')
    inj = moose.Table(f'{data.path}/pulse')
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    vm = moose.Table(f'{data.path}/Vm')
    moose.connect(vm, 'requestOut', soma, 'getVm')
    print('A' * 10, soma)

    simtime = 700e-3
    moose.reinit()
    moose.start(simtime)
    
    t = np.arange(len(vm.vector)) * vm.dt
    print('%' * 10, len(vm.vector), len(inj.vector))
    results = np.block([t[:, np.newaxis], vm.vector[:, np.newaxis], inj.vector[:, np.newaxis]])
    fname = 'Granule_98.dat'
    np.savetxt(fname, X=results, header='time Vm Im', delimiter=' ')
    print(f'Saved results in {fname}')
    if not nogui:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, sharex='all')
        axes[0].plot(t, vm.vector, label='Vm')
        axes[1].plot(t, inj.vector, label='Im')
        plt.legend()
        plt.show()
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} modeldir')
        print('where modeldir contains the neuroML2 model. For example, for GranuleCell model cloned from OpenSourceBrain, this should be `GranuleCell/neuroConstruct/generatedNeuroML2/`')
        sys.exit(1)
    model_dir = sys.argv[1]
    run(model_dir, nogui=False)


# 
# test_granule98.py ends here
