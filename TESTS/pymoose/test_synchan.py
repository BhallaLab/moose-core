# test_synchan.py --- 
# 
# Filename: test_synchan.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Dec 29 14:32:23 2009 (+0530)
# Version: 
# Last-Updated: Tue Dec 29 17:12:21 2009 (+0530)
#           By: subhasis ray
#     Update #: 43
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This is for testing SynChan objects in PyMoose
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

import moose
context = moose.PyMooseBase.getContext()

class TestSynChan:
    def __init__(self):
	self.dendrite = moose.Compartment('/dendrite')
	self.soma = moose.Compartment('/soma')

	self.synchan = moose.SynChan('/dendrite/synchan')
	self.synchan.Ek = 0.0
	self.synchan.gmax = 1e-8
	self.tau1 = 1e-3
	self.tau2 = 2e-3
	# The soma will cause the SpikeGenerator to send out a spike
	# as its Vm goes above threshold.
	self.spikegen = moose.SpikeGen('/soma/spike')
	self.spikegen.threshold = 0.0
	self.spikegen.absRefract = 0.02	
	self.soma.connect('VmSrc', self.spikegen, 'Vm')
	# The SynChan acts as a conductance channel on the dendrite.
# 	self.dendrite.connect('VmSrc', self.synchan, 'Vm')
	self.synchan.connect('channel', self.dendrite, 'channel')
	# Connect the Spike output of the soma to the input of the
	# SynChan on denrite. This will also add an entry to the list
	# of synapses inside the SynChan object.
	self.spikegen.connect('event', self.synchan, 'synapse') 
	# Now that we have a synaptic connection, we can get and set
	# its parameters. The index of this synapse is 0.
	print self.synchan.getWeight(0)
	print self.synchan.getDelay(0)
	self.synchan.setWeight(0.5, 0)
	print self.synchan.getWeight(0)
	self.synchan.setDelay(1.0e-3, 0)
	print self.synchan.getDelay(0)

if __name__ == '__main__':
    testObject = TestSynChan()
# 
# test_synchan.py ends here
