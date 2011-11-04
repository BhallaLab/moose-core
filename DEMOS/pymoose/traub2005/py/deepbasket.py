# deepbasket.py --- 
# 
# Filename: deepbasket.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 14:30:33 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:17:29 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 74
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
    chan_params = {
        'ENa': 50e-3,
        'EK': -100e-3,
        'EAR': -40e-3,
        'ECa': 125e-3,
        'EGABA': -75e-3, # Sanchez-Vives et al. 1997 
        'TauCa': 20e-3,
        'X_AR': 0.25
        }
    ca_dep_chans = ['KC_FAST']
    num_comp = 59
    presyn = 59
    level = TraubCell.readlevels('DeepBasket.levels')
    depth = None
    proto_file = 'DeepBasket.p'
    prototype = TraubCell.read_proto(proto_file, "DeepBasket", chan_params)
    def __init__(self, *args):
        TraubCell.__init__(self, *args)
        moose.CaConc(self.soma.path + '/CaPool').tau = 50e-3
	
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
		    config.LOGGER.debug(comp.name + ' : ' + channel.name + ' connected to ' + ca_pool.name)
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")
		    config.LOGGER.debug(comp.name + ' : ' + ca_pool.name + ' connected to ' + channel.name)

	obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 50e-3


    @classmethod
    def test_single_cell(cls):
        """Simulates a single deep basket cell and plots the Vm and [Ca2+]"""
        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
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
        kc_path = mycell.soma.path + '/KC'
        gk_table = None
        if config.context.exists(kc_path):
            gk_table = moose.Table('gkc', sim.data)
            gk_table.stepMode = 3
            kc = moose.HHChannel(kc_path)
            kc.connect('Gk', gk_table, 'inputRequest')
            pymoose.showmsg(ca_conc)
        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=50e-3, firstWidth=50e-3)
#         pulsegen1 = mycell.soma.insertPulseGen('pulsegen1', sim.model, firstLevel=3e-7, firstDelay=150e-3, firstWidth=10e-3)

        sim.schedule()
        if mycell.has_cycle():
            print "WARNING!! CYCLE PRESENT IN CICRUIT."
        t1 = datetime.now()
        sim.run(200e-3)
        t2 = datetime.now()
        delta = t2 - t1
        print 'simulation time: ', delta.seconds + 1e-6 * delta.microseconds
        sim.dump_data('data')
        if config.has_pylab:
            mus_vm = config.pylab.array(vm_table) * 1e3
            nrn_vm = config.pylab.loadtxt('../nrn/mydata/Vm_deepbask.plot')
            nrn_t = nrn_vm[:, 0]
            mus_t = linspace(0, nrn_t[-1], len(mus_vm))
            nrn_vm = nrn_vm[:, 1]
            nrn_ca = config.pylab.loadtxt('../nrn/mydata/Ca_deepbask.plot')
            nrn_ca = nrn_ca[:,1]
            config.pylab.plot(nrn_t, nrn_vm, 'y-', label='nrn vm')
            config.pylab.plot(mus_t, mus_vm, 'g-.', label='mus vm')
    #         if ca_table:
    #             ca_array = config.pylab.array(ca_table)
    #             config.pylab.plot(nrn_t, -nrn_ca, 'r-', label='nrn (-)ca')
    #             config.pylab.plot(mus_t, -ca_array, 'b-.', label='mus (-)ca')
    #             print config.pylab.amax(ca_table)
            config.pylab.legend()
            config.pylab.show()
        
        
# test main --
from simulation import Simulation
import pylab
from subprocess import call
if __name__ == "__main__":
    DeepBasket.test_single_cell()



# 
# deepbasket.py ends here
