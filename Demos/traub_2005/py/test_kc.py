# test_kc.py --- 
# 
# Filename: test_kc.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sun Jun  3 19:13:25 2012 (+0530)
# Version: 
# Last-Updated: Sun Jun  3 19:59:07 2012 (+0530)
#           By: subha
#     Update #: 11
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


def setup_cadep_channel(container_path, channel_proto, Gbar, ca_start, ca_stop):
    """Setup a test compartment with [Ca2+] dependent channel."""
    params = setup_single_compartment(container_path, channel_proto, Gbar)
    ca_table = moose.StimulusTable(container_path + '/CaStim')    
    ca_table.vec = np.linspace(ca_start, ca_stop, 1000)
    ca_table.doLoop = True
    ca_recorder = moose.Table(container_path + '/Ca')
    moose.connect(ca_table, 'output', ca_recorder, 'input')
    moose.connect(ca_table, 'output', params['channel'], 'concen')
    params['Ca'] = ca_recorder
    params['CaStim'] = ca_table
    moose.useClock(1, '%s,%s' % (ca_recorder.path, ca_table.path), 'process')
    return params

def run_cadep_channel(channelname, Gbar, simtime):
    testId = uuid.uuid4().int
    container = moose.Neutral('test%d' % (testId))
    params = setup_cadep_channel(
        container.path,
        channelbase.prototypes[channelname],
        Gbar,
        0,
        500.0)
    ca_table = params['CaStim']
    ca_table.startTime = 0.0
    ca_table.stopTime = 175e-3
    vm_data = params['Vm']
    gk_data = params['Gk']
    ik_data = params['Ik']
    ca_data = params['Ca']
    moose.reinit()
    print 'Starting simulation', testId, 'for', simtime, 's'
    moose.start(simtime)
    print 'Finished simulation'
    vm_file = 'data/%s_Vm.dat' % (channelname)
    gk_file = 'data/%s_Gk.dat' % (channelname)
    ik_file = 'data/%s_Ik.dat' % (channelname)
    ca_file = 'data/%s_Ca.dat' % (channelname)
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
    np.savetxt(ca_file, data)
    print 'Saved [Ca2+] in', ca_file
    return params

        
class TestKC(ChannelTestBase):
    channelname = 'KC'
    params = run_cadep_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Gk'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKC_Vm_Neuron(self):        
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, self.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        
    def testKC_Gk_Neuron(self):        
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, self.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.01)
        
    

        
if __name__ == '__main__':
    unittest.main()



# 
# test_kc.py ends here
