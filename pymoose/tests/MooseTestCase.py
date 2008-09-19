#!/usr/bin/env python
# Author: Subhasis Ray 
# Created: 2008-09-18 17:44:39 IST
from math import *
import unittest

from moose import *

class MooseTestCase(unittest.TestCase):
    testId = 0
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.testContainer = Neutral("/test")
        self.dataContainer = Neutral("/testData")
        
