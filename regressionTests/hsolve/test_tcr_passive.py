# test_tcr_passive.py --- 
# 
# Filename: test_tcr_passive.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue Aug 28 11:20:55 2012 (+0530)
# Version: 
# Last-Updated: Tue Aug 28 11:56:13 2012 (+0530)
#           By: subha
#     Update #: 115
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
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import sys
import os
import uuid
import numpy as np
import unittest
import pylab

sys.path.append('../../python')
os.environ['NUMPTHREADS'] = '1'

import moose

simdt = 1e-5
simtime = 300e-3

INITCLOCK = 0
ELECCLOCK = 1
CHANCLOCK = 2
POOLCLOCK = 3
LOOKUPCLOCK = 6
STIMCLOCK = 7
PLOTCLOCK = 8

def setup_model(container_path, inject_amplitude, inject_delay, inject_duration):
    container = moose.Neutral(container_path)
    cell = moose.loadModel('TCR.p', container_path + '/TCR')
    pulsegen = moose.PulseGen(container_path+'/pulse')
    pulsegen.level[0] = inject_amplitude
    pulsegen.width[0] = inject_duration
    pulsegen.delay[0] = inject_delay
    pulsegen.delay[1] = 1e9
    soma = moose.element(cell.path+'/comp_1')
    axon = moose.element(cell.path+'/comp_135')
    moose.connect(pulsegen, 'outputOut', soma, 'injectMsg')
    soma_Vm = moose.Table(container_path+'/soma_Vm')
    moose.connect(soma_Vm, 'requestData', soma, 'get_Vm')
    axon_Vm = moose.Table(container_path+'/axon_Vm')
    moose.connect(axon_Vm, 'requestData', axon, 'get_Vm')
    return {'cell': cell,
            'pulsegeb': pulsegen,
            'soma': soma,
            'axon': axon,
            'Vm_soma': soma_Vm,
            'Vm_axon': axon_Vm}


class TestPassiveTCR(unittest.TestCase):
    def setUp(self):
        self.euler_container_path = 'ForwardEuler'
        self.inject_amplitude = 1e-9
        self.inject_delay = 100e-3
        self.inject_duration = 100e-3
        self.euler_dict = setup_model(self.euler_container_path, 
                                      self.inject_amplitude, 
                                      self.inject_delay,
                                      self.inject_duration)
        self.hsolve_container_path = 'HSolve'
        self.hsolve_dict = setup_model(self.hsolve_container_path, 
                                      self.inject_amplitude, 
                                      self.inject_delay,
                                      self.inject_duration)
        self.hsolve = moose.HSolve(self.hsolve_container_path+'/hsolve')
        self.hsolve.dt = simdt
        self.hsolve.target = self.hsolve_dict['cell'].path

        moose.setClock(INITCLOCK, simdt)
        moose.setClock(ELECCLOCK, simdt)
        moose.setClock(CHANCLOCK, simdt)
        moose.setClock(POOLCLOCK, simdt)
        moose.setClock(LOOKUPCLOCK, simdt)
        moose.setClock(STIMCLOCK, simdt)
        moose.setClock(PLOTCLOCK, simdt)
        moose.useClock(INITCLOCK, 
                       self.euler_container_path+'/##[TYPE=Compartment]',
                       'init')
        moose.useClock(ELECCLOCK, 
                       self.euler_container_path+'/##[TYPE=Compartment]',
                       'process')
        moose.useClock(STIMCLOCK, 
                       self.euler_container_path+'/##[TYPE=PulseGen]',
                       'process')
        moose.useClock(PLOTCLOCK, 
                       self.euler_container_path+'/##[TYPE=Table]',
                       'process')
        # Assign clocks to the HSolve test model
        moose.useClock(INITCLOCK, 
                       self.hsolve.path,
                       'process')
        moose.useClock(STIMCLOCK, 
                       self.hsolve_container_path+'/##[TYPE=PulseGen]',
                       'process')
        moose.useClock(PLOTCLOCK, 
                       self.hsolve_container_path+'/##[TYPE=Table]',
                       'process')
        moose.reinit()
        moose.start(simtime)

    def testByVmPlots(self):
        pylab.plot(self.euler_dict['Vm_soma'].vec, label='Vm - Forward Euler method')
        pylab.plot(self.hsolve_dict['Vm_soma'].vec, label='Vm - Hsolve method')
        pylab.legend()
        pylab.show()
    
            
if __name__ == '__main__':
    unittest.main()
        
    
    
# 
# test_tcr_passive.py ends here
