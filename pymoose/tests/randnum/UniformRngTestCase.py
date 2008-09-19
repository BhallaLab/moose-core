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
        self.testId = self.newTestId()        
        self.testObj = UniformRng("UniformRng_" + str(self.testId), self.testContainer)
        self.recordField("sample")
        
    def testSetGet(self):
        """Test getters and setters"""
        mean = 0.5
        variance = 1.0/12
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 1, "Mean differs from default: found " + str(self.testObj.mean) + ", expected: " + str(mean))
        self.failUnlessAlmostEqual(self.testObj.variance, variance, 1, "Variance differs from default: found " + str (self.testObj.variance) + ", expected: " + str(variance))
        min = -1.0
        self.testObj.min = min
        self.failUnlessAlmostEqual((self.testObj.min - min)/min, 0, 1, "Failed to set lower bound")
        max = 1.0
        self.testObj.max = max
        self.failUnlessAlmostEqual((self.testObj.max - max)/max, 0, 1, "Failed to set upper bound")
        mean = (max + min)/2.0
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 1, "Mean differs: found: " + str(mean) + ", expected: " + str(self.testObj.mean))
        variance = (max - min) ** 2 / 12.0
        self.failUnlessAlmostEqual((self.testObj.variance - variance)/variance, 0, 1, "Variance differs: found " + str(variance) +  ", expected " + str(self.testObj.variance))
        
    def testMeanVariance(self):
        """Test if sample mean and variance are close enough to generators's."""
        self.testObj.min = 1.0
        self.testObj.max = 11.0
        mean, variance, = self.sampleMeanVariance()
        self.failUnlessAlmostEqual((self.testObj.mean - mean) / max(self.testObj.mean, mean),
                                   0.0, 1, "Sample mean is " + str(mean) + ", expected ~ " + str(self.testObj.mean))
        self.failUnlessAlmostEqual((self.testObj.variance - variance) / max(self.testObj.variance, variance),
                                   0.0, 1, "Sample variance is " + str(variance) + ", expected ~ " + str(self.testObj.variance))
        
        

if __name__ == "__main__":
    unittest.main()
