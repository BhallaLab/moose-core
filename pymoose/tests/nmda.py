# nmda.py --- 
# 
# Filename: nmda.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Mar 17 14:07:20 2010 (+0530)
# Version: 
# Last-Updated: Wed Mar 17 18:57:29 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 97
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

import pylab
import moose

CONTEXT = moose.PyMooseBase.getContext()
SIMDT = 1e-5
PLOTDT = 1e-5
SIMTIME = 100e-3
# This is not really a serious test - just checking that it gives some
# output.
class TestNMDAChan(object):
    """Test NMDAChan - the NMDA channel as described by Jahr and
    Stevens, 1990.

    Create a compartment soma and attach an NMDA channel called nmda
    to it. Connect a random spike generator to the NMDA channel and
    record the spikes, the nmda conductance and the Vm of soma.
    """
    def __init__(self):
	self.model = moose.Neutral('/model')
	self.data = moose.Neutral('/data')
	# Create soma 
	self.soma = moose.Compartment('soma', self.model)
	self.soma.Rm = 5e9
	self.soma.Ra = 1e5
	self.soma.Cm = 10e-12
	self.soma.Em = -0.065
	self.soma.initVm = -0.065
	# Create the channel
	self.nmda = moose.NMDAChan('nmda', self.soma)
	self.nmda.Gbar = 1e-3 # Siemens
	self.nmda.Mg = 1.5
	self.nmda.tau1 = 0.005
	self.nmda.tau2 = 0.005
	print 'TAU1 = ', self.nmda.tau1, 'TAU2 = ', self.nmda.tau2
	self.nmda.connect('channel', self.soma, 'channel')
	# Create the source of random spikes
	self.spike = moose.RandomSpike('spike', self.model)
	self.spike.rate = 100.0
	self.spike.minAmp = 1.0
	self.spike.maxAmp = 1.0
	self.spike.reset = 1
	self.spike.resetValue = 0.0
	self.spike.absRefract = 2e-3 # second
	self.spike.connect('event', self.nmda, 'synapse')
	# Create tables for recording data
	self.vmTable = moose.Table('Vm', self.data)
	self.gkTable = moose.Table('Gk', self.data)
	self.spikeTable = moose.Table('Spike', self.data)
	self.vmTable.stepMode = 3
	self.gkTable.stepMode = 3
	self.spikeTable.stepMode = 3
	self.vmTable.connect('inputRequest', self.soma, 'Vm')
	self.gkTable.connect('inputRequest', self.nmda, 'Gk')
	self.spikeTable.connect('inputRequest', self.spike, 'state')
	# Setup scheduling
	CONTEXT.setClock(0, SIMDT)
	CONTEXT.setClock(1, SIMDT)
	CONTEXT.setClock(2, PLOTDT)
	CONTEXT.useClock(2, '/data/#[TYPE=Table]')

	
    def testNMDAChan(self):
	CONTEXT.reset()
	CONTEXT.step(SIMTIME)
	# pylab.plot(self.vmTable, 'r-', label='Vm')
	pylab.plot(self.gkTable, 'b-', label='Gk')
	pylab.plot(self.spikeTable, 'k-', label='Spike')
	pylab.legend()
	pylab.show()

if __name__ == '__main__':
    test = TestNMDAChan()
    test.testNMDAChan()

# 
# nmda.py ends here
