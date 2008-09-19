#!/usr/bin/env python
#******************************************************************
# File:            NormalRngTestCase.py
# Description:      
# Author:          Subhasis Ray
# E-mail:          ray dot subhasis at gmail dot com
# Created:         2008-09-18 17:50:12
#*******************************************************************/
#*********************************************************************
# This program is part of 'MOOSE', the
# Messaging Object Oriented Simulation Environment,
# also known as GENESIS 3 base code.
#           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
# It is made available under the terms of the
# GNU General Public License version 2
# See the file COPYING.LIB for the full notice.
#********************************************************************/

from MooseTestCase import *

class NormalRngTestCase(MooseTestCase):
    testId = 0
    def __init__(self, *args):
        MooseTestCase.__init__(self, *args)

    def setUp(self):
        self.testObj = NormalRng("NormalRng_" + str(NormalRngTestCase.testId), self.testContainer)
        self.testData = Table("NormalRngData_" + str(NormalRngTestCase.testId), self.dataContainer)
        self.testData.stepMode = 3
        self.testData.connect("inputRequest", self.testObj, "sample")
        print NormalRngTestCase.testId
        NormalRngTestCase.testId += 1

    def sampleMeanVariance(self, steps=1000):
        self.testObj.getContext().reset()
        self.testObj.getContext().step(steps)
        self.failUnless(len(self.testData) == steps + 1, \
                            "The data table has not been populated correctly.\
 Length is " + str(len(self.testData)) + " , expected" +str(steps + 1))
        self.testData.dumpFile("normalrng.plot")
        mean =  sum(value for value in self.testData)
        mean /= 1001
        variance = sum ( ( value - mean)**2 for value in self.testData )
        variance /= 1001
        return (mean, variance)
        
    def testAliasMethod(self):
        """Testing Box-Mueller method"""
        self.testObj.method = 0
        mean, variance, = self.sampleMeanVariance()
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 1, "Sample mean is " + str(mean) + ", expected ~" + str(self.testObj.mean))
        self.failUnlessAlmostEqual((self.testObj.variance - variance)/max(self.testObj.variance, variance), 0, 0, "Sample variance is " + str(variance) + ", expected ~" + str(self.testObj.variance))

    def testMeanVariance(self):
        """Test if the sample mean and sample variance are sufficiently close to what was set in the generator."""
        self.testObj.mean = 10.0
        self.testObj.variance = 3.0
        mean, variance, = self.sampleMeanVariance()
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 1, "Sample mean is " + str(mean) + ", expected ~" + str(self.testObj.mean))
        self.failUnlessAlmostEqual((self.testObj.variance - variance)/max(self.testObj.variance, variance), 0, 0, "Sample variance is " + str(variance) + ", expected ~" + str(self.testObj.variance))
        

    def testSetGet(self):
        """Test if we can set the fields correctly"""
        mean = 0.0
        variance = 1.0
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 2, "The default mean is not 0.0")
        self.failUnlessAlmostEqual(self.testObj.variance, variance, 2, "The default variance is not 1.0")
        variance = -3.0
        self.testObj.variance = variance
        self.failUnlessAlmostEqual(self.testObj.variance, 1.0, 2, "Did not ignore negative variance")                                   
        mean = 10.0
        self.testObj.mean = mean
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 2, "Could not set mean to " + str(mean))
        variance = 3.0
        self.testObj.variance = variance
        self.failUnlessAlmostEqual(self.testObj.variance, variance, 2, "Could not set variance to " + str(variance))
        
        mean = -10.0
        self.testObj.mean = mean
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 2, "Could not set mean to " + str(mean))
        
        self.testObj.getContext().reset()
        self.failUnlessAlmostEqual(self.testObj.mean, mean, 2, "Mean changed after reset()")
        self.failUnlessAlmostEqual(self.testObj.variance, variance, 2, "Variance changed after reset()")

class NormalRngTestSuite(unittest.TestSuite):
    def __init__(self):
        unittest.TestSuite.__init__(self, map(NormalRngTestCase, ("testSetGet", "testMeanVariance")))
                                   

if __name__=="__main__":
    unittest.main()

