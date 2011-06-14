# test_stpsynchan.py --- 
# 
# Filename: test_stpsynchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Mon Jun 13 14:59:06 2011 (+0530)
# Version: 
# Last-Updated: Tue Jun 14 19:15:33 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 299
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Test for STPSynchan
# 
# 

# Change log:
# 
# 
# 

# Code:

import unittest
import uuid

import moose


class TestSTPSynChan(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.testId = uuid.uuid4().int
        # These are the common parameters to be used in tests
        self.tauF = 1e-3
        self.tauD1 = 2e-3
        self.tauD2 = 20e-3
        self.d1 = 0.8
        self.d2 = 0.3
        self.deltaF = 0.01
        self.initPr = [0.8, 0.3]
        self.initF = [0.9, 0.7]
        self.initD1 = [0.6, 0.4]
        self.initD2 = [0.7, 0.2]
        self.Rm = 1e9
        self.Ra = 1e6
        self.Cm = 1e-9
        self.Em = -70e-3
        self.initVm = -70.0e-3
        self.dt = 1.0e-5
        self.container = moose.Neutral('/testSTPSynChan%d' % (self.testId))
        moose.context.setCwe(self.container.path)
        self.synchan = moose.STPSynChan('syn')

    def setUp(self):
        self.synchan.tauD1 = self.tauD1
        self.synchan.tauD2 = self.tauD2
        self.synchan.tauF = self.tauF
        self.synchan.deltaF = self.deltaF
        self.synchan.d1 = self.d1
        self.synchan.d2 = self.d2

    def setupSynapses(self):
        self.spikegen1 = moose.SpikeGen('spike1')
        self.spikegen2 = moose.SpikeGen('spike2')
        self.spikegen1.connect('event', self.synchan, 'synapse')
        self.assertEqual(self.synchan.numSynapses, 1)        
        self.spikegen2.connect('event', self.synchan, 'synapse')
        for ii in range(self.synchan.numSynapses):
            self.synchan.setInitPr(ii, self.initPr[ii])
            self.synchan.setInitD1(ii, self.initD1[ii])
            self.synchan.setInitD2(ii, self.initD2[ii])
            self.synchan.setInitF(ii, self.initF[ii])
        
    def setupStimulus(self):
        self.pulsegen = moose.PulseGen('pulsegen')
        self.pulsegen.baseLevel = -1.0
        self.pulsegen.firstLevel = 1.0
        self.pulsegen.firstDelay = 2.0 * self.dt
        self.pulsegen.firstWidth = 1.0 * self.dt
        self.pulsegen.secondDelay = 1e9
        self.pulsegen.connect('outputSrc', self.spikegen1, 'Vm')
        self.pulsegen.connect('outputSrc', self.spikegen2, 'Vm')
        self.spikegen1.threshold = 0.0
        self.spikegen2.threshold = 0.0
        
                
    def testSetGet(self):
        self.assertAlmostEqual(self.synchan.tauD1, self.tauD1)
        self.assertAlmostEqual(self.synchan.tauD2, self.tauD2)
        self.assertAlmostEqual(self.synchan.tauF, self.tauF)
        self.assertAlmostEqual(self.synchan.deltaF, self.deltaF)
        self.assertAlmostEqual(self.synchan.d1, self.d1)
        self.assertAlmostEqual(self.synchan.d2, self.d2)

    def testSynapses(self):
        self.setupSynapses()
        self.assertEqual(self.synchan.numSynapses, 2)
        for ii in range(self.synchan.numSynapses):
            self.assertAlmostEqual(self.synchan.getInitPr(ii), self.initPr[ii])
            self.assertAlmostEqual(self.synchan.getInitD1(ii), self.initD1[ii])
            self.assertAlmostEqual(self.synchan.getInitD2(ii), self.initD2[ii])
            self.assertAlmostEqual(self.synchan.getInitF(ii), self.initF[ii])
    
    def testReinit(self):
        self.setupSynapses()
        moose.context.reset()
        for ii in range(self.synchan.numSynapses):
            self.assertAlmostEqual(self.synchan.getF(ii), self.initF[ii])
            self.assertAlmostEqual(self.synchan.getD1(ii), self.initD1[ii])
            self.assertAlmostEqual(self.synchan.getD2(ii), self.initD2[ii])
        
    def testProcess(self):
        self.setupSynapses()
        self.setupStimulus()
        self.soma = moose.Compartment('soma')
        self.soma.Rm = self.Rm
        self.soma.Ra = self.Ra
        self.soma.Cm = self.Cm
        self.soma.Em = self.Em
        self.soma.initVm = self.initVm
        self.assertTrue(self.synchan.connect('channel', self.soma, 'channel'))
        moose.context.setClock(0, self.dt)
        moose.context.setClock(1, self.dt)
        moose.context.setClock(2, self.dt)
        moose.context.setClock(3, self.dt)
        moose.context.reset()
        moose.context.srandom(1)
        moose.context.step(1)
        prev_F = []
        prev_D1 = []
        prev_D2 = []
        for ii in range(self.synchan.numSynapses):
            tmpF = self.initF[ii]
            # We calculate two steps because of this moose bug:
            # step(1) after reset() always does 2 steps
            tmpF = ( 1 - tmpF ) * self.dt / self.tauF + tmpF 
            tmpF = ( 1 - tmpF ) * self.dt / self.tauF + tmpF            
            self.assertAlmostEqual(self.synchan.getF(ii), tmpF)
            tmpD1 = self.initD1[ii]
            tmpD1 = ( 1 - tmpD1 ) * self.dt / self.tauD1 + tmpD1
            tmpD1 = ( 1 - tmpD1 ) * self.dt / self.tauD1 + tmpD1            
            self.assertAlmostEqual(self.synchan.getD1(ii), tmpD1)
            tmpD2 = self.initD2[ii]
            tmpD2 = ( 1 - tmpD2 ) * self.dt / self.tauD2 + tmpD2
            tmpD2 = ( 1 - tmpD2 ) * self.dt / self.tauD2 + tmpD2            
            self.assertAlmostEqual(self.synchan.getD2(ii), tmpD2)
            prev_F.append(self.synchan.getF(ii))
            prev_D1.append(self.synchan.getD1(ii))
            prev_D2.append(self.synchan.getD2(ii))
        moose.context.step(2)
        for ii in range(self.synchan.numSynapses):
            tmpF = self.deltaF + prev_F[ii]
            tmpF = (1 - tmpF) * self.dt / self.tauF + tmpF
            tmpF = (1 - tmpF) * self.dt / self.tauF + tmpF
            self.assertAlmostEqual(self.synchan.getF(ii), tmpF)
            tmpD1 = self.d1 * prev_D1[ii]
            tmpD1 = (1 - tmpD1) * self.dt / self.tauD1 + tmpD1
            tmpD1 = (1 - tmpD1) * self.dt / self.tauD1 + tmpD1
            self.assertAlmostEqual(self.synchan.getD1(ii), tmpD1)
            tmpD2 = self.d2 * prev_D2[ii]
            tmpD2 = (1 - tmpD2) * self.dt / self.tauD2 + tmpD2
            tmpD2 = (1 - tmpD2) * self.dt / self.tauD2 + tmpD2
            self.assertAlmostEqual(self.synchan.getD2(ii), tmpD2)
        
if __name__ == '__main__':
    unittest.main()

# 
# test_stpsynchan.py ends here
