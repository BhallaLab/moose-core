# suppyrFRB.py --- 
# 
# Filename: suppyrFRB.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Sep 21 01:45:00 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:10:23 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 129
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
    chan_params = {
    'ENa': 50e-3,
    'EK': -95e-3,
    'EAR': -35e-3,
    'ECa': 125e-3,
    'EGABA': -81e-3,
    'TauCa': 20e-3
    }
    ca_dep_chans = ['KAHP', 'KC']
    num_comp = 74
    presyn = 72
    proto_file = 'SupPyrFRB.p'
    # level maps level number to the set of compartments belonging to it
    level = None
    # depth stores a map between level number and the depth of the compartments.
    depth = {
        1: 850 * 1e-6,
        2: 885 * 1e-6,
        3: 920 * 1e-6,
        4: 955 * 1e-6,
        5: 825 * 1e-6,
        6: 775 * 1e-6,
        7: 725 * 1e-6,
        8: 690 * 1e-6,
        9: 655 * 1e-6,
        10: 620 * 1e-6,
        11: 585 * 1e-6,
        12: 550 * 1e-6,
        }
    prototype = TraubCell.read_proto(proto_file, "SupPyrFRB", level_dict=level, depth_dict=depth, params=chan_params)

    def __init__(self, *args):
        TraubCell.__init__(self, *args)
        soma_ca_pool = moose.CaConc(self.soma.path + '/CaPool')
        soma_ca_pool.tau = 100e-3

    def _topology(self):
        raise Exception, 'Deprecated'
	self.presyn = 72

    def _setup_passive(self):
        raise Exception, 'Deprecated'
	for comp in self.comp[1:]:
	    comp.initVm = -70e-3

    def _setup_channels(self):
        raise Exception, 'Deprecated'
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
		    config.LOGGER.debug(comp.name + ':' + channel.name + ' connected to ' + ca_pool.name)
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")
		    config.LOGGER.debug(comp.name + ': ' + ca_pool.name + ' connected to ' + channel.name)

	obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 100e-3


    @classmethod
    def test_single_cell(cls):
        """Simulates a single superficial pyramidal FRB cell and plots
        the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
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
            mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))
            try:
                nrn_vm = config.pylab.loadtxt('../nrn/mydata/Vm_deepLTS.plot')
                nrn_t = nrn_vm[:, 0]
                nrn_vm = nrn_vm[:, 1]
                config.pylab.plot(nrn_t, nrn_vm, 'y-', label='nrn vm')
            except IOError:
                print 'NEURON Data not available.'
            config.pylab.plot(mus_t, mus_vm, 'g-.', label='mus vm')
            config.pylab.legend()
            config.pylab.show()
        
        
# test main --
from simulation import Simulation
import pylab
from subprocess import call

if __name__ == "__main__":
    SupPyrFRB.test_single_cell()
    



# 
# suppyrFRB.py ends here
