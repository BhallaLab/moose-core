# squid_demo.py --- 
# 
# Filename: squid_demo.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Feb 22 23:24:21 2012 (+0530)
# Version: 
# Last-Updated: Tue Mar  6 15:35:51 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 110
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

import moose

from squid import SquidAxon
from electronics import ClampCircuit

class SquidDemo(object):
    def __init__(self):
        self.scheduled = False
        self.model_container = moose.Neutral('/model')
        self.data_container = moose.Neutral('/data')
        self.squid_axon = SquidAxon('/model/squid_axon')
        self.clamp_ckt = ClampCircuit('/model/electronics', self.squid_axon)
        self.simdt = 0.0
        self.plotdt = 0.0
        self.vm_table = moose.Table('/data/vm')
        self.vm_table.stepMode = 3
        moose.connect(self.vm_table, 'requestData', self.squid_axon, 'get_Vm')        
        
    def schedule(self, simdt, plotdt):
        self.simdt = simdt
        self.plotdt = plotdt
        moose.setClock(0, simdt)
        moose.setClock(1, simdt)
        moose.setClock(2, simdt)
        moose.setClock(3, plotdt)
        # Ensure we do not create multiple scheduling
        if not self.scheduled:
            moose.useClock(0, '%s/#[TYPE=Compartment]' % (self.model_container.path), 'init')
            moose.useClock(0, '%s/##' % (self.clamp_ckt.path), 'process')
            moose.useClock(1, '%s/#[TYPE=Compartment]' % (self.model_container.path), 'process')
            moose.useClock(2, '%s/#[TYPE=HHChannel]' % (self.squid_axon.path), 'process')
            moose.useClock(3, '%s/#[TYPE=Table]' % (self.data_container.path), 'process')
            self.scheduled = True
        moose.reinit()
        
    def run(self, runtime, clamp_mode='vclamp'):
        if clamp_mode == 'vclamp':
            self.clamp_ckt.do_voltage_clamp()
            self.schedule(0.01, 0.1)
        else:
            self.clamp_ckt.do_current_clamp()
            self.schedule(0.1, 0.1)
        moose.start(runtime)

    def save_data(self):
        for child in self.data_container.children:
            tab = moose.Table(child)
            tab.xplot('%s.dat' % (tab.name), tab.name)

import sys            
clamp_mode = 'vclamp'
if __name__ == '__main__':
    demo = SquidDemo()
    if len(sys.argv) > 1:
        clamp_mode = sys.argv[1]
    # The pulsegen object needs different values for different clamp modes
    if clamp_mode == 'iclamp':
        demo.clamp_ckt.configure_pulses()
    else:
        demo.clamp_ckt.configure_pulses(baselevel=0.0, firstdelay=10.0, firstlevel=SquidAxon.EREST_ACT, firstwidth=0.0, seconddelay=0.0, secondlevel=50.0+SquidAxon.EREST_ACT, secondwidth=20.0)
    # demo.schedule(1e-2, 0.01)
    
    demo.run(50.0, clamp_mode)
    demo.save_data()

# 
# squid_demo.py ends here
