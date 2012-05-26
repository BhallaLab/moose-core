# test_nachans.py --- 
# 
# Filename: test_nachans.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat May 26 10:29:41 2012 (+0530)
# Version: 
# Last-Updated: Sat May 26 16:13:57 2012 (+0530)
#           By: subha
#     Update #: 98
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Unit tests for single Na channels
# 
# 

# Change log:
# 
# 
# 
# 

# Code:

import sys
sys.path.append('../../../python')
import numpy as np
import unittest
import moose
import init
import nachans
from testutils import setup_single_compartment, compare_data_arrays

simtime = 1.0

class TestNaF(unittest.TestCase):
    def setUp(self):
        init.init()
        container = moose.Neutral('testNaF')
        self.params = setup_single_compartment(
            container.path,
            nachans.initNaChannelPrototypes()['NaF'])
        self.vm_data = self.params['Vm']
        self.gk_data = self.params['Gk']
        self.ref_vm_data = np.loadtxt('testdata/NaF_Vm.dat.gz')
        self.ref_gk_data = np.loadtxt('testdata/NaF_Gk.dat.gz')
        moose.reinit()
        # moose.showfield(self.params['channel'])
        print 'Starting simulation for', simtime, 's'
        moose.start(simtime)
        print 'Saving xplot data in\n NaF_Vm.dat\n NaF_Gk.dat'
        self.vm_data.xplot('NaF_Vm.dat', 'Vm')
        self.gk_data.xplot('NaF_Gk.dat', 'Gk')
        print 'Finished simulation'
        
    def testNaF(self):
        err = compare_data_arrays(self.ref_vm_data, np.array(self.vm_data.vec), plot=True)
        # The relative error is of the order of 1e-4
        self.assertAlmostEqual(err, 0.0, places=3)
        err = compare_data_arrays(self.ref_gk_data, np.array(self.gk_data.vec))
        self.assertAlmostEqual(err, 0.0, places=3)

if __name__ == '__main__':
    unittest.main()
    
#
# test_nachans.py ends here
