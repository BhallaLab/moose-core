#!/usr/bin/env python

# /*******************************************************************
#  * File:            test_bulbchan.py
#  * Description:      Unittest for bulbchan
#  * Author:          Subhasis Ray
#  * E-mail:          ray dot subhasis at gmail dot com
#  * Created:         2008-10-23 11:01:41
#  ********************************************************************/
# /**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Messaging Object Oriented Simulation Environment,
# ** also known as GENESIS 3 base code.
# **           copyright (C) 2008 Upinder S. Bhalla. and NCBS
# ** It is made available under the terms of the
# ** GNU General Public License version 2
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/

import math
import unittest

import moose

import bulbchan



class TestKMitralUSB(unittest.TestCase):
    """Create a compartment with one KMitralUSB channel and test with a current pulse."""
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.container = moose.Neutral("/TestKMitralUSB")
        self.data = moose.Neutral("data", self.container)
        self.sim_dt = 50e-6
        self.io_dt = 50e-6
        self.sim_length = 0.05
        self.inject = 5e-10
        self.erest_act = -0.065

        moose.PyMooseBase.getContext().setClock(0, self.sim_dt, 0)
        moose.PyMooseBase.getContext().setClock(1, self.sim_dt, 0)
        moose.PyMooseBase.getContext().setClock(2, self.io_dt, 0)
        
    def setUp(self):
        pass

    def testChannelCurrent(self):
        # Set up a compartment for testing
        compartment = moose.Compartment("compartment" , self.container)
        compartment.length = 28e-6
        compartment.diameter = 19e-6
        s_area = math.pi * compartment.diameter * compartment.length
        compartment.Rm = 2.0 / s_area
        compartment.Cm = 0.01 * s_area
        x_area = math.pi * compartment.diameter * compartment.diameter / 4.0
        compartment.Ra = 0.5 * compartment.length / x_area
        compartment.Ek = self.erest_act
        compartment.inject = 0.0
        print "Ra =", compartment.Ra, "Rm =", compartment.Rm, "Cm =", compartment.Cm
        channel = bulbchan.KMitralUSB("K_channel", compartment)
        channel.Ik = 0.0
        channel.connect("channel", compartment, "channel")
        print "Gbar =", channel.Gbar

        pulse = moose.PulseGen("inject", self.container)
        pulse.firstLevel = self.inject
        pulse.firstWidth = 0.01
        pulse.firstDelay = 0.01
#        pulse.trigMode = 0
        pulse.connect("outputSrc", compartment, "injectMsg")

        channel_plot = moose.Table("KMitralUSB_Ik", self.data)
        channel_plot.connect("inputRequest", channel, "Ik")
        channel_plot.stepMode = 3
        channel_plot.useClock(2)

        inject_plot = moose.Table("KMitralUSB_inject", self.data)
        inject_plot.connect("inputRequest", pulse, "output")
        inject_plot.stepMode = 3
        inject_plot.useClock(2)
        
        vm_plot = moose.Table("KMitralUSB_Vm", self.data)
        vm_plot.connect("inputRequest", compartment, "Vm")
        vm_plot.stepMode = 3
        vm_plot.useClock(2)

        moose.PyMooseBase.getContext().reset()
        moose.PyMooseBase.getContext().reset()

        moose.PyMooseBase.getContext().step(self.sim_length)
        channel_plot.dumpFile(channel_plot.name + ".pymoose.plot")
        inject_plot.dumpFile(inject_plot.name + ".pymoose.plot")
        vm_plot.dumpFile(vm_plot.name + ".pymoose.plot")
# ! testChannelCurrent
# ! TestKMitralUSB

class TestNaMitralUSB(unittest.TestCase):
    """Create a compartment with one KMitralUSB channel and test with a current pulse."""
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.container = moose.Neutral("/TestNaMitralUSB")
        self.data = moose.Neutral("data", self.container)
        self.sim_dt = 50e-6
        self.io_dt = 50e-6
        self.sim_length = 0.05
        self.inject = 5e-10
        self.erest_act = -0.065

        moose.PyMooseBase.getContext().setClock(0, self.sim_dt, 0)
        moose.PyMooseBase.getContext().setClock(1, self.sim_dt, 0)
        moose.PyMooseBase.getContext().setClock(2, self.io_dt, 0)
        
    def setUp(self):
        pass

    def testChannelCurrent(self):
        # Set up a compartment for testing
        compartment = moose.Compartment("compartment" , self.container)
        compartment.length = 28e-6
        compartment.diameter = 19e-6
        s_area = math.pi * compartment.diameter * compartment.length
        compartment.Rm = 2.0 / s_area
        compartment.Cm = 0.01 * s_area
        x_area = math.pi * compartment.diameter * compartment.diameter / 4.0
        compartment.Ra = 0.5 * compartment.length / x_area
        compartment.Ek = self.erest_act
        channel = bulbchan.NaMitralUSB("Na_channel", compartment)
        channel.connect("channel", compartment, "channel")
        
        pulse = moose.PulseGen("inject", self.container)
        pulse.firstLevel = self.inject
        pulse.firstWidth = 0.01
        pulse.firstDelay = 0.01
        pulse.trigMode = 0
        pulse.connect("outputSrc", compartment, "injectMsg")

        channel_plot = moose.Table("NaMitralUSB_Ik", self.data)
        channel_plot.connect("inputRequest", channel, "Ik")
        channel_plot.stepMode = 3
        channel_plot.useClock(2)
        inject_plot = moose.Table("NaMitralUSB_inject", self.data)
        inject_plot.connect("inputRequest", pulse, "output")
        inject_plot.stepMode = 3
        inject_plot.useClock(2)  
        moose.PyMooseBase.getContext().reset()
        moose.PyMooseBase.getContext().step(self.sim_length)
        # channel_plot.dumpFile(channel_plot.name + ".pymoose.plot")
        # inject_plot.dumpFile(inject_plot.name + ".pymoose.plot")

if __name__ == "__main__":
    unittest.main()
