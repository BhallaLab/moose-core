# suppyrFRB.py --- 
# 
# Filename: suppyrFRB.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Sep 21 01:45:00 2009 (+0530)
# Version: 
# Last-Updated: Sat Oct 17 10:17:38 2009 (+0530)
#           By: subhasis ray
#     Update #: 114
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
import string
from datetime import datetime
import config
import trbutil
import moose
from cell import *
from capool import CaPool

class SupPyrFRB(TraubCell):
    prototype = TraubCell.read_proto("SupPyrFRB.p", "SupPyrFRB")
    def __init__(self, *args):
        self.chan_list = ['all channels']
#         self.chan_list = ['NaF', 'NaP', 'KC', 'KAHP', 'KDR', 'KA', 'K2', 'KM', 'CaL', 'CaT', 'AR']
	TraubCell.__init__(self, *args)


    def _topology(self):
	self.presyn = 72

    def _setup_passive(self):
	for comp in self.comp[1:]:
	    comp.initVm = -70e-3

    def _setup_channels(self):
	for comp in self.comp[1:]:
	    ca_pool = None
	    ca_dep_chans = []
	    ca_chans = []
	    for child in comp.children():
		obj = moose.Neutral(child)
		if obj.name == 'CaPool':
		    ca_pool = moose.CaConc(child)
		    ca_pool.tau = 20e-3
		else:
		    obj_class = obj.className
		    if obj_class == 'HHChannel':
			obj = moose.HHChannel(child)
#                         if not obj.name in self.chan_list:
#                             obj.Gbar = 0.0
			pyclass = eval(obj.name)
			if issubclass(pyclass, KChannel):
			    obj.Ek = -95e-3
			    if issubclass(pyclass, KCaChannel):
				ca_dep_chans.append(obj)
			elif issubclass(pyclass, NaChannel):
			    obj.Ek = 50e-3
			elif issubclass(pyclass, CaChannel):
			    obj.Ek = 125e-3
			    if issubclass(pyclass, CaL):
				ca_chans.append(obj)
			elif issubclass(pyclass, AR):
			    obj.Ek = -35e-3
	    if ca_pool:
		for channel in ca_chans:
		    channel.connect('IkSrc', ca_pool, 'current')
		    print comp.name, ':', channel.name, 'connected to', ca_pool.name
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")
		    print comp.name, ':', ca_pool.name, 'connected to', channel.name

	obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 100e-3


    @classmethod
    def test_single_cell(cls):
        """Simulates a single superficial pyramidal FRB cell and plots
        the Vm and [Ca2+]"""

        print "/**************************************************************************"
        print " *"
        print " * Simulating a single cell: ", cls.__name__
        print " *"
        print " **************************************************************************/"
        sim = Simulation()
        mycell = SupPyrFRB(SupPyrFRB.prototype, sim.model.path + "/SupPyrFRB")
        print 'Created cell:', mycell.path
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_suppyrFRB', 'Vm', sim.data)
        ca_conc_path = mycell.soma.path + '/CaPool'
        ca_table = None
        if config.context.exists(ca_conc_path):
            ca_conc = moose.CaConc(ca_conc_path)
            ca_table = moose.Table('Ca_suppyrFRB', sim.data)
            ca_table.stepMode = 3
            ca_conc.connect('Ca', ca_table, 'inputRequest')
        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=50e-3, firstWidth=50e-3)


        sim.schedule()
        if mycell.has_cycle():
            print "WARNING!! CYCLE PRESENT IN CICRUIT."
        t1 = datetime.now()
        sim.run(200e-3)
        t2 = datetime.now()
        delta = t2 - t1
        print 'simulation time: ', delta.seconds + 1e-6 * delta.microseconds
        sim.dump_data('data')
        mycell.dump_cell('suppyrFRB.txt')
        mus_vm = pylab.array(vm_table) * 1e3
        mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))
        ca_array = pylab.array(ca_table)

        if config.neuron_bin:
            call([config.neuron_bin, 'test_suppyrFRB.hoc'], cwd='../nrn')

        nrn_vm = trbutil.read_nrn_data('Vm_suppyrFRB.plot')
        nrn_t = nrn_vm[:, 0]
        nrn_vm = nrn_vm[:, 1]
        nrn_ca = trbutil.read_nrn_data('Ca_suppyrFRB.plot')
        nrn_ca = nrn_ca[:,1]
        
        pylab.subplot(211)
        pylab.plot(nrn_t, nrn_vm, 'y-', label='NEURON')
        pylab.plot(mus_t, mus_vm, 'g-.', label='MOOSE')
        pylab.title('Vm in presynaptic compartment of %s' % cls.__name__)
        pylab.legend()
        pylab.subplot(212)
        pylab.plot(nrn_t, nrn_ca, 'r-', label='NEURON')
        pylab.plot(mus_t, ca_array, 'b-.', label='MOOSE')
        pylab.title('[Ca2+] in soma of %s' % cls.__name__)
        pylab.legend()
        pylab.show()
        
        
# test main --
from simulation import Simulation
import pylab
from subprocess import call

if __name__ == "__main__":
    SupPyrFRB.test_single_cell()
    



# 
# suppyrFRB.py ends here
