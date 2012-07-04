# test_nap.py --- 
# 
# Filename: test_nap.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed May 30 20:00:50 2012 (+0530)
# Version: 
# Last-Updated: Wed May 30 20:01:13 2012 (+0530)
#           By: subha
#     Update #: 4
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

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')
import numpy as np

from testutils import *
import nachans


class TestNaP(ChannelTestBase):
    channelname = 'NaP'
    params = run_single_channel(channelname, 1e-9, simtime)
    moose.showfield(params['channel'])
    vm = np.asarray(params['Vm'].vec)        
    gk = np.asarray(params['Gk'].vec)
    tseries = np.array(range(0, len(params['Vm'].vec))) * simdt
    def testNaP_Vm_Moose(self):
        print 'Testing MOOSE Vm  ...',
        err = compare_channel_data(self.vm, TestNaP.channelname, 'Vm', 'moose', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'
        
    def testNaP_Gk_Moose(self):
        print 'Testing MOOSE Gk  ...',
        err = compare_channel_data(self.gk, TestNaP.channelname, 'Gk', 'moose', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.05)
        print 'OK'

    def testNaP_Vm_Neuron(self):
        print 'Testing NEURON Vm  ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, self.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)        
        print 'OK'

    def testNaP_Gk_Neuron(self):
        print 'Testing NEURON Gk  ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, self.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.05)
        print 'OK'



# 
# test_nap.py ends here
