# test_kchans.py --- 
# 
# Filename: test_kchans.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed May 30 23:51:58 2012 (+0530)
# Version: 
# Last-Updated: Thu May 31 00:52:59 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 40
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

import numpy as np
from testutils import *
from kchans import *

class TestKDR(ChannelTestBase):
    channelname = 'KDR'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKDR_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKDR.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKDR_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKDR.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

        
class TestKDR_FS(ChannelTestBase):
    channelname = 'KDR_FS'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKDR_FS_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKDR_FS.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKDR_FS_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKDR_FS.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

        
class TestKA(ChannelTestBase):
    channelname = 'KA'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKA_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKA.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKA_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKA.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'


class TestKA_IB(ChannelTestBase):
    channelname = 'KA_IB'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKA_IB_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKA_IB.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKA_IB_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKA_IB.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

        
class TestK2(ChannelTestBase):
    channelname = 'K2'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testK2_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestK2.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testK2_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestK2.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

        
class TestKM(ChannelTestBase):
    channelname = 'KM'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Vm'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKM_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKM.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKM_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKM.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

        
        
if __name__ == '__main__':
    unittest.main()
    
# 
# test_kchans.py ends here
