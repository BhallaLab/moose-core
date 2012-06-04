# test_capool.py --- 
# 
# Filename: test_capool.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sun Jun  3 20:31:03 2012 (+0530)
# Version: 
# Last-Updated: Mon Jun  4 11:55:12 2012 (+0530)
#           By: subha
#     Update #: 60
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

import uuid
import numpy as np
from testutils import *
from cachans import *
from capool import *


def run_capool(poolname, Gbar, simtime):
    testId = uuid.uuid4().int
    container = moose.Neutral('test%d' % (testId))
    params = setup_single_compartment(
        container.path,
        channelbase.prototypes['CaL'],
        Gbar)
    channelname = 'CaL'
    capool = moose.copy(CaPool.prototype, params['compartment'], 'CaPool')[0]
    moose.connect(params['channel'], 'IkOut', capool, 'current')
    # The B is obtained from phi in NEURON by dividing it with
    # compartment area in cm2 and multiplying by 1e3 for /mA->/A and
    # by 1e3 for /ms->/S
    capool.B = 52000 * 1e6 / (3.141592 * 1e-4 * 1e-4)
    // beta = 1/tau (ms) = 0.02 => tau = 50 ms
    capool.tau = 50e-3
    ca_data = moose.Table('%s/Ca' % (container.path))
    moose.connect(ca_data, 'requestData', capool, 'get_Ca')
    moose.useClock(2, '%s,%s' % (capool.path, ca_data.path), 'process')
    vm_data = params['Vm']
    gk_data = params['Gk']
    ik_data = params['Ik']
    params['Ca'] = ca_data
    moose.reinit()
    print 'Starting simulation', testId, 'for', simtime, 's'
    moose.start(simtime)
    print 'Finished simulation'
    vm_file = 'data/%s_Vm.dat' % (poolname)
    gk_file = 'data/%s_Gk.dat' % (poolname)
    ik_file = 'data/%s_Ik.dat' % (poolname)
    ca_file = 'data/%s_Ca.dat' % (poolname)
    tseries = np.array(range(len(vm_data.vec))) * simdt
    print 'Vm:', len(vm_data.vec), 'Gk', len(gk_data.vec), 'Ik', len(ik_data.vec)
    data = np.c_[tseries, vm_data.vec]
    np.savetxt(vm_file, data)
    print 'Saved Vm in', vm_file
    data = np.c_[tseries, gk_data.vec]
    np.savetxt(gk_file, data)
    print 'Saved Gk in', gk_file
    data = np.c_[tseries, ik_data.vec]
    np.savetxt(ik_file, data)
    print 'Saved Ik in', ik_file
    data = np.c_[tseries, ca_data.vec]
    np.savetxt(ca_file, data)
    print 'Saved [Ca2+] in', ca_file
    return params
    

class TestCaPool(ChannelTestBase):
    channelname = 'CaL'
    poolname = 'CaPool'    
    params = run_capool(poolname, 1e-9, 350e-3)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Gk'].vec)
    ca = np.array(params['Ca'].vec)
    print len(ca)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testCaPool_Vm_Neuron(self):
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, self.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)

    def testCaPool_Gk_Neuron(self):
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, self.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.01)
        
    def testCaPool_Ca_Neuron(self):
        print self.ca.shape
        data = np.c_[self.tseries, self.ca]
        err = compare_channel_data(data, self.poolname, 'Ca', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.01)

if __name__ == '__main__':
    unittest.main()

# 
# test_capool.py ends here
