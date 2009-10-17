# deepbasket.py --- 
# 
# Filename: deepbasket.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 14:30:33 2009 (+0530)
# Version: 
# Last-Updated: Sat Oct 17 02:46:57 2009 (+0530)
#           By: subhasis ray
#     Update #: 62
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

from datetime import datetime
import config
import moose
import trbutil
from cell import *
from capool import CaPool


class DeepBasket(TraubCell):
    prototype = TraubCell.read_proto("DeepBasket.p", "DeepBasket")
    ca_dep_chans = ['KAHP','KAHP_SLOWER', 'KAHP_DP', 'KC', 'KC_FAST']
    def __init__(self, *args):
	TraubCell.__init__(self, *args)
	
    def _topology(self):
        self.presyn = 59
    
    def _setup_passive(self):
        for comp in self.comp[1:]:
	    comp.initVm = -65e-3

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
	unblock = ['KDR_FS', 'NaF2', 'CaL', 'KA', 'KC_FAST']
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
			if obj.name not in unblock:
			    obj.Gbar = 0.0
			pyclass = eval(obj.name)
			if issubclass(pyclass, KChannel):
			    obj.Ek = -100e-3
			    if issubclass(pyclass, KCaChannel):
				ca_dep_chans.append(obj)
			elif issubclass(pyclass, NaChannel):
			    obj.Ek = 50e-3
			elif issubclass(pyclass, CaChannel):
			    obj.Ek = 125e-3
			    if issubclass(pyclass, CaL):
				ca_chans.append(obj)
			elif issubclass(pyclass, AR):
			    obj.Ek = -40e-3
	    if ca_pool:
		for channel in ca_chans:
		    channel.connect('IkSrc', ca_pool, 'current')
		    print comp.name, ':', channel.name, 'connected to', ca_pool.name
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")
		    print comp.name, ':', ca_pool.name, 'connected to', channel.name

	obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 50e-3


    @classmethod
    def test_single_cell(cls):
        """Simulates a single deep basket cell and plots the Vm and [Ca2+]"""
        print "/**************************************************************************"
        print " *"
        print " * Simulating a single cell: ", cls.__name__
        print " *"
        print " **************************************************************************/"
        sim = Simulation()
        mycell = DeepBasket(DeepBasket.prototype, sim.model.path + "/DeepBasket")
        print 'Created cell:', mycell.path
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_deepbask', 'Vm', sim.data)
        ca_conc_path = mycell.soma.path + '/CaPool'
        ca_table = None
        if config.context.exists(ca_conc_path):
            ca_conc = moose.CaConc(ca_conc_path)
            ca_table = moose.Table('Ca_deepbask', sim.data)
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
        mus_vm = pylab.array(vm_table) * 1e3
        ca_array = pylab.array(ca_table)
        mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))

        if config.neuron_bin:
            call([config.neuron_bin, 'test_deepbask.hoc'], cwd='../nrn')
            
        nrn_vm = trbutil.read_nrn_data('Vm_deepbask.plot')
        nrn_t = nrn_vm[:, 0]
        nrn_vm = nrn_vm[:, 1]
        nrn_ca = trbutil.read_nrn_data('Ca_deepbask.plot')
        nrn_ca = nrn_ca[:,1]
        pylab.subplot(2,1,1)
        pylab.plot(nrn_t, nrn_vm, 'y-', label='NEURON')
        pylab.plot(mus_t, mus_vm, 'g-.', label='MOOSE')
        pylab.title('Vm of presynaptic compartment of %s cell' % cls.__name__)
        pylab.legend()
        pylab.subplot(2,1,2)
        pylab.plot(nrn_t, nrn_ca, 'r-', label='NEURON')
        pylab.plot(mus_t, ca_array, 'b-.', label='MOOSE')
        pylab.title('[Ca2+] of soma compartment of %s cell' % cls.__name__)
        pylab.legend()
        pylab.show()
        
        
# test main --
from simulation import Simulation
import pylab
from subprocess import call
if __name__ == "__main__":
    DeepBasket.test_single_cell()



# 
# deepbasket.py ends here
