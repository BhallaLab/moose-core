#*******************************************************************
#* File:            RandNumTest.py
#* Description:     Common utility functions for testing rngs.
#* Author:          Subhasis Ray
#* E-mail:          ray dot subhasis at gmail dot com
#* Created:         2008-09-19 14:22:13
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
from MooseTestCase import *

class RandNumTestCase(MooseTestCase):
    def __init__(self, *args):
        MooseTestCase.__init__(self, *args)

    def sampleMeanVariance(self, testObj=None, testData=None, steps=1000):
        if testObj is None:
            testObj = self.testObj
        if testData is None:
            testData = self.testData
        testObj.getContext().reset()
        print testObj.name, "- Mean: ", testObj.mean, ", Variance:", testObj.variance
        testObj.getContext().step(steps)
        self.failUnless(len(testData) == steps + 1, \
                       "The data table has not been populated correctly.\
 Length is " + str(len(testData)) + " , expected" +str(steps + 1))
        testData.dumpFile("normalrng.plot")
        mean =  sum(value for value in testData)
        mean /= (steps + 1)
        variance = sum ( ( value - mean)**2 for value in testData )
        variance /= (steps + 1)
        return (mean, variance)

