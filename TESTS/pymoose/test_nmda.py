# nmda.py --- 
# 
# Filename: nmda.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Mar 17 14:07:20 2010 (+0530)
# Version: 
# Last-Updated: Tue Jun 21 17:07:22 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 291
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import unittest
import uuid
import moose

has_numpy = True

try:
    import numpy
except ImportError:
    has_numpy = False

pi = 3.141592

class TestNMDAChan(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.testId = 0
        # simulation settings
        self.simdt = 1e-5
        self.simtime = 0.5
        # compartment properties
        self.dia = 15e-6
        self.len = 20e-6
        self.ra = 250.0 * 1e-2
        self.g_pas = 2e-5 * 1e4
        self.e_pas = -65e-3
        self.cm = 0.9e-6 * 1e4

        # stimulus for presynaptic compartment
        self.stim_amp = 0.1e-9
        self.stim_dur = 20e-3
        self.stim_delay = 20e-3

        # parameters for the NMDA channel
        self.tau1 = 130.5e-3
        self.tau2 = 5e-3
        self.MgConc = 1.5
        self.saturation = 0.25 # NMDA_saturation_fact. This is multiplied with weight of the NMDA object (equivalent to Gbar in MOOSE.
        
        self.NMDA_weight = 0.25e-3 * 1e-6 # uS->S : This is the weight specified for the NetConn object in NEURON.
        self.Gbar = 1.0e-4 # Not to be used except for calculating saturation
        
    def setUp(self):
        self.testId = uuid.uuid4().int
        self.container = moose.Neutral('testNMDA_%d' % (self.testId))
        self.nmda = moose.NMDAChan('nmda', self.container)
        self.nmda_gk = moose.Table('nmda_gk', self.container)
        self.nmda_gk.stepMode = 3
        self.nmda_gk.connect('inputRequest', self.nmda, 'Gk')
        self.nmda_unblocked = moose.Table('nmda_unblocked', self.container)
        self.nmda_unblocked.stepMode = 3
        self.nmda_unblocked.connect('inputRequest', self.nmda, 'unblocked')
        
        
    def setParameters(self):
        self.nmda.tau1 = self.tau1
        self.nmda.tau2 = self.tau2
        self.nmda.Gbar = self.Gbar
        self.nmda.MgConc = self.MgConc
        self.nmda.saturation = self.saturation
        self.nmda.Gbar = self.Gbar
        
    def testSetGet(self):
        self.setParameters()
        self.assertAlmostEqual(self.nmda.tau1, self.tau1)
        self.assertAlmostEqual(self.nmda.tau2, self.tau2)
        self.assertAlmostEqual(self.nmda.Gbar, self.Gbar)
        self.assertAlmostEqual(self.nmda.MgConc, self.MgConc)
        self.assertAlmostEqual(self.nmda.saturation, self.saturation)

    def setupNetwork(self):
        self.setParameters()
        self.somaA = moose.Compartment('a', self.container)
        self.somaA.Rm = 1.0 /(self.g_pas * self.len * self.dia * pi)
        self.somaA.Ra = self.ra / (self.dia * self.dia * pi / 4.0)
        self.somaA.Cm = self.cm * self.len * self.dia * pi
        self.somaA.Em = self.e_pas
        self.somaA.initVm = self.e_pas
        self.pulsegen = moose.PulseGen('pulsegen', self.container)
        self.pulsegen.firstLevel = self.stim_amp
        self.pulsegen.firstDelay = self.stim_delay
        self.pulsegen.firstWidth = self.stim_dur
        self.pulsegen.secondDelay = 1e9
        self.pulsegen.connect('outputSrc', self.somaA, 'injectMsg')
        self.somaB = moose.Compartment('b', self.container)
        self.somaB.Rm = 1.0 /(self.g_pas * self.len * self.dia * pi)
        self.somaB.Ra = self.ra / (self.dia * self.dia * pi / 4.0)
        self.somaB.Cm = self.cm * self.len * self.dia * pi
        self.somaB.Em = self.e_pas
        self.somaB.initVm = self.e_pas
        self.vmA = moose.Table('Vm_A', self.container)
        self.vmA.stepMode = 3
        self.vmA.connect('inputRequest', self.somaA, 'Vm')
        self.vmB = moose.Table('Vm_B', self.container)
        self.vmB.stepMode = 3
        self.vmB.connect('inputRequest', self.somaB, 'Vm')
        
        self.nmda.connect('channel', self.somaB, 'channel')
        self.spikegen = moose.SpikeGen('spike', self.container)
        self.spikegen.threshold = 0.0
        self.spikegen.delay = 0.05e-3
        self.spikegen.edgeTriggered = 1
        self.spikegen.connect('event', self.nmda, 'synapse')
        print 'Connecting spikegen:', self.somaA.connect('VmSrc', self.spikegen, 'Vm')
        self.assertEqual(self.nmda.numSynapses, 1)
        self.nmda.setWeight(0, self.NMDA_weight)
        self.assertAlmostEqual(self.NMDA_weight, self.nmda.getWeight(0))
        
    def testGkChanges(self):
        self.setupNetwork()
        moose.context.setClock(0, self.simdt)
        moose.context.setClock(1, self.simdt)
        moose.context.setClock(2, self.simdt)
        moose.context.reset()
        moose.context.step(self.simtime)
        outfile = open('moose_nmda.dat', 'w')
        t = 0.0
        tvec = []
        for ii in range(len(self.nmda_gk)):
            outfile.write('%g %g %g %g\n' % (t, self.vmA[ii], self.vmB[ii], self.nmda_gk[ii]/self.nmda_unblocked[ii] if self.nmda_unblocked[ii] != 0 else self.nmda_gk[ii])) # We save nmda_gk/nmda_unblocked as neuron does not compute the gk correctly.
            tvec.append(t)
            t += self.simdt
        outfile.close()
        nrn_data = numpy.loadtxt('neuron_nmda.dat').transpose()
        if has_numpy:
            interpolated_vb = numpy.interp(tvec, nrn_data[0]*1e-3, nrn_data[2]*1e-3)
            error_vec = interpolated_vb - numpy.array(self.vmB)
            square_error = error_vec * error_vec
            rms_error =  numpy.sqrt(square_error.sum() / len(error_vec))
            relative_error = rms_error / interpolated_vb.max()
            self.assertTrue(numpy.abs(relative_error) < 1e-3) # We just choose arbitrarily that relative error should be less than a thousandth
        
        
        
if __name__ == '__main__':
    unittest.main()
    
# 
# nmda.py ends here
