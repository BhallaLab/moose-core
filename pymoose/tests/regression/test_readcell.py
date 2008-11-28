# test_readcell.py --- 
# 
# Filename: test_readcell.py
# Description: test readCell command
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Nov 27 18:08:44 2008 (+0530)
# Version: 
# Last-Updated: Fri Nov 28 18:40:09 2008 (+0530)
#           By: Subhasis Ray
#     Update #: 71
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: This test fails due to incompatibility between MOOSE and
# GENESIS for handling "." as parent element. The test assumes GENESIS
# documentation as the standard.
# 
# 
# 

# Change log:
# 
# 
# 
# 
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/


# Code:

import unittest

import moose

class TestReadCell(unittest.TestCase):
    """Test readcell command in PyMOOSE"""
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.context = moose.PyMooseBase.getContext()
        self.container = moose.Neutral("/TestReadCell")
        moose.HHChannel("/library/Na_mit_usb")
        moose.HHChannel("/library/K_mit_usb")
        self.proto_file_name = "cell.p"
        self.create_proto_file()
        self.targetCellPath = self.container.path + "/cell"

    def testReadCell(self):
        """Tests reading the file and constructing the model"""
        self.context.readCell(self.proto_file_name, self.targetCellPath)
        paths = [ "soma", "soma/h1", "soma/c1" ]
        self.failUnless(self.context.exists(self.targetCellPath))
        for item in paths:
            path = self.targetCellPath + "/" + item
            self.failUnless(self.context.exists(path), "path: " + path + " was not created")
            comp = moose.Compartment(path)
            self.__setattr__(comp.name, comp)
            self.failUnless(self.context.exists(path + "/Na_mit_usb"), "path: " + path + " was not created")
            chan = moose.HHChannel(path + "/Na_mit_usb")
            comp.__setattr__(chan.name, chan)
            self.failUnless(self.context.exists(path + "/K_mit_usb"), "path: " + path + " was not created")
            chan = moose.HHChannel(path + "/Na_mit_usb")
            comp.__setattr__(chan.name, chan)
            
    def create_proto_file(self):
        """Write a fixed .p file in order to be read by readcell.
        
        The idea is to make this test script self contained by keeping the test data in the test program itself. Otherwise there is a tendency for the test input file and the test script to drift away."""
        self.test_input = """
// coarse asymmetrical mitral cell (olf bulb) model : camit
// Control lines start with '*'. Valid control options are 
// *relative 			- relative coords.
// *absolute			- absolute coords.
// *asymmetric			- use asymmetric compartments
// *symmetric			- use symmetric compartments

// #	name	parent		x	y	z	d	ch	dens	ch	dens...
// *hsolve

*symmetric
*relative

*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01

soma		none		0	0	28	19	Na_mit_usb	2e3	K_mit_usb	4e3
h1		.		0	0	10	2	Na_mit_usb	2e3	K_mit_usb	4e3
c1		.		0	0	10	1	Na_mit_usb	2e3	K_mit_usb	4e3
"""
        out_file = open(self.proto_file_name,"w")
        out_file.write(self.test_input)
        out_file.close()

if __name__ == "__main__":
    unittest.main()
# 
# readcell.py ends here
