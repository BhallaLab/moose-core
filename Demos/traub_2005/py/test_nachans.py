# test_nachans.py --- 
# 
# Filename: test_nachans.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat May 26 10:29:41 2012 (+0530)
# Version: 
# Last-Updated: Tue May 29 17:39:36 2012 (+0530)
#           By: subha
#     Update #: 280
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
import config

lib = moose.Neutral(config.modelSettings.libpath)

import channelbase
import nachans
from testutils import *

simtime = 350e-3


class ChannelTestBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def compare_Vm(self, filename):
        print 'Comparing Vm ...'
        self.ref_vm_data = np.loadtxt(filename)
        err = compare_data_arrays(self.ref_vm_data, np.array(self.vm_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'

    def compare_Gk(self, filename):
        print 'Comparing Gk ...'
        self.ref_gk_data = np.loadtxt(filename)
        err = compare_data_arrays(self.ref_gk_data, np.array(self.gk_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'
        
        
class TestNaF(ChannelTestBase):
    channelname = 'NaF'
    params = run_single_channel(channelname, 1e-9, simtime)
    tseries = np.array(range(0, len(params['Vm'].vec))) * simdt
    print  'Gbar', params['channel'].Gbar
    print 'Gk_last', params['Gk'].vec[-1]
    def testNaF_Vm_Moose(self):
        vm = np.asarray(self.params['Vm'].vec)        
        err = compare_channel_data(vm, TestNaF.channelname, 'Vm', 'moose')
        self.assertAlmostEqual(err, 0.0)
        
    def testNaF_Gk_Moose(self):
        gk = np.asarray(self.params['Gk'].vec)
        err = compare_channel_data(gk, TestNaF.channelname, 'Gk', 'moose')
        self.assertAlmostEqual(err, 0.0)

    def testNaF_Vm_Neuron(self):
        vm = np.asarray(self.params['Vm'].vec)
        data = np.c_[self.tseries, vm]
        err = compare_channel_data(data, self.channelname, 'Vm', 'neuron')
        self.assertAlmostEqual(err, 0.0)        

    # def testNaF_Gk_Neuron(self):
    #     gk = np.asarray(self.params['Gk'].vec)
    #     data = np.c_[self.tseries, gk]
    #     err = compare_channel_data(data, self.channelname, 'Gk', 'neuron')
    #     self.assertAlmostEqual(err, 0.0)

    def testNaF_Ik_Neuron(self):
        ik = np.asarray(self.params['Ik'].vec)
        data = np.c_[self.tseries, ik]
        err = compare_channel_data(data, self.channelname, 'Ik', 'neuron')
        self.assertAlmostEqual(err, 0.0)
                

if __name__ == '__main__':
    unittest.main()
    
#
# test_nachans.py ends here
