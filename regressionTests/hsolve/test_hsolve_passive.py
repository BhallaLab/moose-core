# test_hsolve_single_comp.py --- 
# 
# Filename: test_hsolve_single_comp.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue Jul 10 16:16:55 2012 (+0530)
# Version: 
# Last-Updated: Wed Jul 11 10:09:12 2012 (+0530)
#           By: subha
#     Update #: 218
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

import uuid
import unittest
import numpy as np
import matplotlib.pyplot as plt
from  hsolvetestutil import *

def setup_single_comp_model(model_container, data_container, simdt):
    cell = moose.Neuron(model_container.path + '/neuron')
    soma = make_testcomp(cell.path)
    pulsegen = make_pulsegen(model_container.path)
    pulsegen.firstDelay = simdt
    moose.connect(pulsegen, 'outputOut', soma, 'injectMsg')
    vm_table = moose.Table(data_container.path + '/Vm')
    moose.connect(vm_table, 'requestData', soma, 'get_Vm')
    return {'pulsegen': pulsegen,
            'soma': soma,
            'cell': cell,
            'vm_table': vm_table}

def setup_hsolve_single_comp_model(model_container, data_container, simdt):
    elements = setup_single_comp_model(model_container, data_container, simdt)
    solver = moose.HSolve(model_container.path + '/hsolve')
    solver.dt = simdt
    solver.target = elements['cell'].path
    elements['solver'] = solver
    return elements

        
    
class TestSingleCompPassive(unittest.TestCase):
    """Test hsolve with a single compartment"""
    def setUp(self):
        self.simdt = 1e-4
        self.testId = uuid.uuid4().int
        test_container = moose.Neutral('test%d' % (self.testId))
        hsolve_data = moose.Neutral('%s/hsolvedata' % (test_container.path))
        hsolve_model = moose.Neutral('%s/hsolvemodel' % (test_container.path))
        self.hsolve_elements = setup_hsolve_single_comp_model(hsolve_model, hsolve_data, self.simdt)
        fwdeuler_model = moose.Neutral('%s/fwdeulermodel' % (test_container.path))
        fwdeuler_data = moose.Neutral('%s/fwdeulerdata' % (test_container.path))
        self.fwdeuler_elements = setup_single_comp_model(fwdeuler_model, fwdeuler_data, self.simdt)
        run_simulation(test_container, self.simdt, 1000*self.simdt)
        print self.hsolve_elements['vm_table'].vec
        print self.fwdeuler_elements['vm_table'].vec

    def testHSolveSingleComp(self):
        err = compare_data_arrays(self.hsolve_elements['vm_table'].vec, self.fwdeuler_elements['vm_table'].vec, plot=True)
        print 'testHSolveSingleComp: Error:', err
        self.assertLess(err, 0.01)

        
if __name__ == '__main__':
    unittest.main()


# 
# test_hsolve_single_comp.py ends here
