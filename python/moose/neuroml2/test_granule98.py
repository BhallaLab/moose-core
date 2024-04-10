# test_granule98.py --- 
# 
# Filename: test_granule98.py
# Description: 
# Author: Subhasis Ray
# Created: Mon Apr  8 21:41:22 2024 (+0530)
# Last-Updated: Wed Apr 10 19:59:48 2024 (+0530)
#           By: Subhasis Ray
# 

# Code:
"""Test code for the Granule cell model

"""
import os
import numpy as np
# import unittest
import logging


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(level=LOGLEVEL)


import moose
from moose.neuroml2.reader import NML2Reader


def run(nogui=True):
    reader = NML2Reader()
    filename = 'test_files/Granule_98/GranuleCell.net.nml'    
    reader.read(filename)
    soma = reader.getComp(reader.doc.networks[0].populations[0].id, 0, 0)
    data = moose.Neutral('/data')
    pg = reader.getInput('Gran_10pA')
    inj = moose.Table(f'{data.path}/pulse')
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    vm = moose.Table(f'{data.path}/Vm')
    moose.connect(vm, 'requestOut', soma, 'getVm')
    print('A' * 10, soma)

    simtime = 300e-3
    moose.reinit()
    moose.start(simtime)
    
    t = np.arange(len(vm.vector)) * vm.dt
    print('%' * 10, len(vm.vector), len(inj.vector))
    results = np.array([t, vm.vector, inj.vector], dtype=[('time', float), ('Vm', float), ('Im', float)])
    fname = 'Granule_98.npy'
    np.save(fname, results)
    print(f'Saved results in {fname}')
    if not nogui:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, sharex='all')
        axes[0].plot(t, vm.vector, label='Vm')
        axes[1].plot(t, inj.vector, label='Im')
        plt.legend()
        plt.show()
        
    
run(nogui=False)


# 
# test_granule98.py ends here
