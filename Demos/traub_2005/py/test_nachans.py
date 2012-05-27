# test_nachans.py --- 
# 
# Filename: test_nachans.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat May 26 10:29:41 2012 (+0530)
# Version: 
# Last-Updated: Sun May 27 17:27:24 2012 (+0530)
#           By: subha
#     Update #: 159
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

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')
import uuid
import numpy as np
import unittest
import moose
import init
from channelinit import init_chanlib
import nachans
from testutils import setup_single_compartment, compare_data_arrays

simtime = 1.0

class ChannelTestBase(unittest.TestCase):
    init.init()
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.testId = uuid.uuid4().int
        self.container = moose.Neutral('test%d' % (self.testId))
        self.params = setup_single_compartment(
            self.container.path,
            init_chanlib()[self.__class__.channel_name])
        self.vm_data = self.params['Vm']
        self.gk_data = self.params['Gk']
        moose.reinit()
        moose.showfield(self.params['channel'])
        print 'Starting simulation for', simtime, 's'
        moose.start(simtime)
        print 'Finished simulation'
        vm_file = '%s_Vm.dat' % self.__class__.channel_name
        gk_file = '%s_Gk.dat' % self.__class__.channel_name
        np.savetxt(vm_file, np.asarray(self.vm_data.vec))
        print 'Saved Vm in', vm_file
        np.savetxt(gk_file, np.asarray(self.gk_data.vec))
        print 'Saved Gk in', gk_file

    def compare_moose_Vm(self):
        print 'Comparing Vm ...'
        self.ref_vm_data = np.loadtxt('testdata/%s_Vm.dat.gz' % (self.__class__.channel_name))
        err = compare_data_arrays(self.ref_vm_data, np.array(self.vm_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'

    def compare_moose_Gk(self):
        print 'Comparing Gk ...'
        self.ref_gk_data = np.loadtxt('testdata/%s_Gk.dat.gz' % (self.__class__.channel_name))
        err = compare_data_arrays(self.ref_gk_data, np.array(self.gk_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'
        
        
class TestNaF(ChannelTestBase):
    channel_name = 'NaF'
    def testNaF_Vm_Moose(self):
        self.compare_moose_Vm()
        
    def testNaF_Vm_Moose(self):
        self.compare_moose_Gk()
        

if __name__ == '__main__':
    unittest.main()
    
#
# test_nachans.py ends here
