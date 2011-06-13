# test_stpsynchan.py --- 
# 
# Filename: test_stpsynchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Mon Jun 13 14:59:06 2011 (+0530)
# Version: 
# Last-Updated: Mon Jun 13 17:36:15 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 81
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

import moose


class TestSTPSynChan(unittest.TestCase):
    def setUp(self):
        self.container = moose.Neutral('/testSTPSynChan')
        moose.context.setCwe(self.container.path)
        self.soma = moose.Compartment('soma')
        self.synchan = moose.STPSynChan('syn')
        self.soma.connect('channel', self.synchan, 'channel')
        self.pulsegen = moose.PulseGen('pulsegen')
        self.spikegen1 = moose.SpikeGen('spike1')
        self.spikegen2 = moose.SpikeGen('spike2')
        self.pulsegen.connect('outputSrc', self.spikegen1, 'Vm')
        self.pulsegen.connect('outputSrc', self.spikegen1, 'Vm')
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

    def testNumSynapse(self):
        self.spikegen1.connect('event', self.synchan, 'synapse')
        self.assertEqual(self.synchan.numSynapses, 1)        
        self.spikegen2.connect('event', self.synchan, 'synapse')
        self.assertEqual(self.synchan.numSynapses, 2)

    def testSetGet(self):
        self.synchan.tauD1 = self.tauD1
        self.assertAlmostEqual(self.synchan.tauD1, self.tauD1)
        self.synchan.tauD2 = self.tauD2
        self.assertAlmostEqual(self.synchan.tauD2, self.tauD2)
        self.synchan.tauF = self.tauF
        self.assertAlmostEqual(self.synchan.tauF, self.tauF)
        self.synchan.deltaF = self.deltaF
        self.assertAlmostEqual(self.synchan.deltaF, self.deltaF)
        self.synchan.d1 = self.d1
        self.assertAlmostEqual(self.synchan.d1, self.d1)
        self.synchan.d2 = self.d2
        self.assertAlmostEqual(self.synchan.d2, self.d2)
        for ii in range(self.synchan.numSynapses):
            self.synchan.setInitPr(ii, self.initPr[ii])
            self.assertAlmostEqual(self.synchan.getInitPr(ii), self.initPr[ii])
            self.synchan.setInitD1(ii, self.initD1[ii])
            self.assertAlmostEqual(self.synchan.getInitD1(ii), self.initD1[ii])
            self.synchan.setInitD2(ii, self.initD2[ii])
            self.assertAlmostEqual(self.synchan.getInitD2(ii), self.initD2[ii])
            self.synchan.setInitF(ii, self.initF[ii])
            self.assertAlmostEqual(self.synchan.getInitF(ii), self.initF[ii])

    def testReinit(self):
        self.synchan.tauD1 = self.tauD1
        self.assertAlmostEqual(self.synchan.tauD1, self.tauD1)
        self.synchan.tauD2 = self.tauD2
        self.assertAlmostEqual(self.synchan.tauD2, self.tauD2)
        self.synchan.tauF = self.tauF
        self.assertAlmostEqual(self.synchan.tauF, self.tauF)
        self.synchan.deltaF = self.deltaF
        self.assertAlmostEqual(self.synchan.deltaF, self.deltaF)
        self.synchan.d1 = self.d1
        self.assertAlmostEqual(self.synchan.d1, self.d1)
        self.synchan.d2 = self.d2
        self.assertAlmostEqual(self.synchan.d2, self.d2)
        for ii in range(self.synchan.numSynapses):
            self.synchan.setInitPr(ii, self.initPr[ii])
            self.assertAlmostEqual(self.synchan.getInitPr(ii), self.initPr[ii])
            self.synchan.setInitD1(ii, self.initD1[ii])
            self.assertAlmostEqual(self.synchan.getInitD1(ii), self.initD1[ii])
            self.synchan.setInitD2(ii, self.initD2[ii])
            self.assertAlmostEqual(self.synchan.getInitD2(ii), self.initD2[ii])
            self.synchan.setInitF(ii, self.initF[ii])
            self.assertAlmostEqual(self.synchan.getInitF(ii), self.initF[ii])
        moose.context.reset()
        for ii in range(self.synchan.numSynapses):
            self.assertAlmostEqual(self.synchan.getF(ii), self.synchan.getInitF(ii))
            self.assertAlmostEqual(self.synchan.getD1(ii), self.synchan.getInitD1(ii))
            self.assertAlmostEqual(self.synchan.getD2(ii), self.synchan.getInitD2(ii))
        
        # TODO: add test for actual simulated values - for the process function.

        
if __name__ == '__main__':
    unittest.main()

# 
# test_stpsynchan.py ends here
