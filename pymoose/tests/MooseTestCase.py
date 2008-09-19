#!/usr/bin/env python
# Author: Subhasis Ray 
# Created: 2008-09-18 17:44:39 IST
from math import *
import unittest

from moose import *

class MooseTestCase(unittest.TestCase):
    __testId = 0
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.testContainer = Neutral("/test")
        self.dataContainer = Neutral("/testData")
        self.testObj = None
        self.testData = None
    
    def newTestId(self):
        MooseTestCase.__testId += 1
        return MooseTestCase.__testId

    def recordField(self, fieldName, tableName=None, object=None):
        """Connect the data table to a particular field of object"""
        data = None
        if tableName is None:
            self.testData = Table(self.testObj.name, self.dataContainer)
            data = self.testData
        else:
            data = Table(tableName, self.dataContainer)
        if object is None:
            object = self.testObj
        data.stepMode = 3
        data.connect("inputRequest", object, fieldName)
        print "Connected", data.name, "to", object.name
        return data
