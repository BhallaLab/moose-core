# efield.py --- 
# 
# Filename: efield.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Jul 21 15:12:50 2011 (+0530)
# Version: 
# Last-Updated: Thu Jul 21 15:28:48 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 38
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
from math import *

class EfieldDemo:
    def __init__(self):
        self.dt = 1e-5
        container = moose.Neutral('demo')
        comp = moose.Compartment('soma', container)
        comp.length = 20e-6
        comp.diameter = 2 * 7.5e-6
        comp.xarea = pi * comp.diameter * comp.diameter / 4.0
        comp.sarea = pi * comp.diameter * comp.length
        comp.Rm = 1.0 / comp.sarea			# specific rm = 1.0 Ohm-m^2
        comp.Ra = 2.5 * comp.length / comp.xarea	# specific ra = 2.5 Ohm-m
        comp.Cm = 1e-3 * comp.sarea			# spcific cm = 1e-3 F/m^2
        comp.Em = -70e-3
        comp.initVm = -70e-3
        pulsegen = moose.PulseGen('pulsegen', container)
        pulsegen.firstLevel = 100e-12	# 100 pA current
        pulsegen.firstWidth = 20e-3		# each pulse 20 ms wide
        pulsegen.firstDelay = 20e-3		# pulses every 20 ms
        pulsegen.connect('outputSrc', comp, 'injectMsg')

        efield = moose.Efield('electrode', container)
        efield.scale = -3.33e4
        efield.x = 100e-6
        efield.y = 0.0
        efield.z = 0.0
        comp.connect('ImSrc', efield, 'currentDest')

        VmTab = moose.Table('Vm', container)
        VmTab.stepMode = 3
        comp.connect('Vm', VmTab, 'inputRequest')

        fieldTab = moose.Table('LFP', container)
        fieldTab.stepMode = 3
        efield.connect('potential', fieldTab, 'inputRequest')

        moose.context.setClock(0, self.dt)
        moose.context.setClock(1, self.dt)
        moose.context.setClock(2, self.dt)

        self.pulsegen = pulsegen
        self.comp = comp
        self.efield = efield
        self.VmTab = VmTab
        self.fieldTab = fieldTab
        
    def run(self):
        vm_file = 'efield_soma_Vm.dat'
        lfp_file = 'efield_LFP.dat'
        moose.context.reset()
        moose.context.step(1.0)
        self.VmTab.dumpFile(vm_file)
        self.fieldTab.dumpFile(lfp_file)
        print 'Finished simulation. soma Vm saved in %s and LFP at 100 u is saved in %s.' % (vm_file, lfp_file)


if __name__ == '__main__':
    demo = EfieldDemo()
    demo.run()
    


# 
# efield.py ends here
