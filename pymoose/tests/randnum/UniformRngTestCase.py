#!/usr/bin/env python
#*******************************************************************
#* File:            UniformRngTestCase.py
#* Description:      
#* Author:          Subhasis Ray
#* E-mail:          ray dot subhasis at gmail dot com
#* Created:         2008-09-19 10:21:51
#********************************************************************/
#**********************************************************************
#* This program is part of 'MOOSE', the
#* Messaging Object Oriented Simulation Environment,
#* also known as GENESIS 3 base code.
#*           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
#* It is made available under the terms of the
#* GNU General Public License version 2
#* See the file COPYING.LIB for the full notice.
#*********************************************************************/

from RandNumTestCase import *

class UniformRngTestCase(RandNumTestCase):
    def __init__(self, *args):
        MooseTestCase.__init__(self, *args)

    def setUp(self):
        self.testObj = UniformRng("UniformRng_" + str(MooseTestCase.testId), self.testContainer)
        self.recordField("sample")
        
    def testSetGet(self):
        mean = 0.5
        variance = 1.0/12
        self.failUnlessAlmostEqual((self.testObj.mean - mean)/mean, 0, 1, "Mean differs from assigned default mean.")
        self.failUnlessAlmostEqual((self.testObj.variance - variance)/variance, 0, 1, "Variance differs from assigned default variance.")
        min = -1.0
        self.testObj.min = min
        self.failUnlessAlmostEqual((self.testObj.min - min)/min, 0, 1, "Failed to set lower bound")
        max = 1.0
        self.testObj.max = max
        self.failUnlessAlmostEqual((self.testObj.max - max)/max, 0, 1, "Failed to set upper bound")
        mean = (max - mean)/2.0
        self.failUnlessAlmostEqual((self.testObj.mean - mean)/mean, 0, 1, "Mean differs from assigned default mean.")
        variance = (max - mean) ** 2 / 12.0
        self.failUnlessAlmostEqual((self.testObj.variance - variance)/variance, 0, 1, "Variance differs from assigned default variance.")
        
                                
