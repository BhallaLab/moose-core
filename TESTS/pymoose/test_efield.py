# test_efield.py --- 
# 
# Filename: test_efield.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Sep 28 12:31:43 2011 (+0530)
# Version: 
# Last-Updated: Wed Sep 28 16:10:11 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 118
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

import unittest
import uuid
import os
import numpy

import moose

comp_pos = [[  1.28151389e-06,   3.28516806e-05,   4.34033507e-05],
            [  3.97458434e-05,   3.13696550e-05,   4.78284191e-06],
            [  7.81225903e-05,   5.52200047e-05,   1.14471225e-05],
            [  1.51851428e-05,   2.90013683e-05,   5.59742276e-05],
            [  6.27772622e-05,   3.17341615e-05,   1.88991378e-05],
            [  6.71202581e-05,   4.73622441e-05,   1.89676926e-05],
            [  8.93229757e-05,   1.98005883e-05,   5.21311088e-05],
            [  1.59775209e-05,   1.90757203e-06,   5.81280477e-05],
            [  9.37027485e-05,   1.05817903e-05,   8.62110626e-05],
            [  2.14710708e-05,   1.81916032e-05,   5.72403065e-05]]

def create_compartment(path, dia=1e-6, length=1e-6, specific_raxial=2.5, specific_conductance=0.2, Em=-65e-3, specific_cm=0.009):
    comp = moose.Compartment(path)
    comp.Rm = 1.0/(specific_conductance * length * dia * numpy.pi)
    comp.Ra = specific_raxial / (dia * dia * numpy.pi/4.0)
    comp.Cm = specific_cm * (length * dia * numpy.pi)
    comp.length = length
    comp.diameter = dia
    comp.Em = Em
    comp.initVm = Em
    return comp

def create_pulsegen(path,
                    firstLevel=100e-12,
                    firstWidth=20e-3,
                    firstDelay=20e-3,
                    secondLevel=100e-12,
                    secondWidth=1e9,
                    secondDelay=1e9):
    pulsegen = moose.PulseGen(path)
    pulsegen.firstLevel = firstLevel
    pulsegen.firstWidth = firstWidth
    pulsegen.firstDelay = firstDelay
    pulsegen.secondLevel = secondLevel
    pulsegen.secondWidth = secondWidth
    pulsegen.secondDelay = secondDelay
    return pulsegen
    

    
class TestEfield(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.simdt = 1e-5
        self.simtime = 0.5
        self.test_id = None
        self.model_container = moose.Neutral('/test')
        self.data_container = moose.Neutral('/data')
        self.data_dir = 'efield_data'
        if not os.access(self.data_dir, os.W_OK):
            os.mkdir(self.data_dir)
                
            
        
    def setUp(self):
        self.test_id = uuid.uuid4().int
        self.efield = moose.Efield('electrode%d' % (self.test_id), self.model_container)
        self.efield.scale = -3.33e4
        self.efield.x = 100e-6
        self.efield.y = 0.0
        self.efield.z = 0.0
        self.lfp_table = moose.Table('lfp%d' % (self.test_id), self.data_container)
        self.lfp_table.stepMode = 3
        self.efield.connect('potential', self.lfp_table, 'inputRequest')
        
    def testMultiCompartments(self):
        global comp_pos
        vm_tabs = []
        comps = []
        pulsegens = []
        numcomps = 10
        for ii in range(numcomps):
            comps.append(create_compartment('%s/test%d_comp_%d' % (self.model_container.path, self.test_id, ii)))
            comps[-1].x = comp_pos[ii][0]
            comps[-1].y = comp_pos[ii][1]
            comps[-1].z = comp_pos[ii][2]
            comps[-1].connect('ImSrc', self.efield, 'currentDest')
            pulsegens.append(create_pulsegen('%s/test%d_pulse_%d' % (self.model_container.path, self.test_id, ii), firstDelay=ii*20e-3))
            pulsegens[-1].connect('outputSrc', comps[-1], 'injectMsg')
            vm_tabs.append(moose.Table('%s/vm%d_%d' % (self.data_container.path, self.test_id, ii)))
            vm_tabs[-1].stepMode = 3
            comps[-1].connect('Vm', vm_tabs[-1], 'inputRequest')
        moose.context.setClock(0, self.simdt)
        moose.context.setClock(1, self.simdt)
        moose.context.setClock(2, self.simdt)
        moose.context.setClock(3, self.simdt)
        moose.context.reset()        
        moose.context.step(self.simtime)
        self.lfp_table.dumpFile('%s/%s.dat' % (self.data_dir, self.lfp_table.name))
        for tab in vm_tabs:
            tab.dumpFile('%s/%s.dat' % (self.data_dir, tab.name))
        print 'LFP saved in %s', (self.lfp_table.name + '.dat')
        
if __name__ == '__main__':
    unittest.main()
            
        


# 
# test_efield.py ends here
