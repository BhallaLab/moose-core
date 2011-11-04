# tcr.py --- 
# 
# Filename: tcr.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 10:14:07 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:14:04 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 60
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: This is a redoing of the Thalamocortical relay cells using prototype file.
# It is a translation of the cell in Traub et al, 2005 model.
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
import trbutil
import moose
from cell import *
from capool import CaPool


class TCR(TraubCell):
    chan_params = {
        'ENa': 50e-3,
        'EK': -95e-3,
        'EAR': -35e-3,
        'ECa': 125e-3,
        'EGABA': -75e-3, # Sanchez-Vives et al. 1997 
        'TauCa': 20e-3,
        'X_AR': 0.25
    }
    ca_dep_chans = ['KAHP_SLOWER', 'KC']
    num_comp = 137
    presyn = 135
    level = None
    depth = None
    proto_file = 'TCR.p'
    prototype = TraubCell.read_proto(proto_file, "TCR", chan_params)
    def __init__(self, *args):
        TraubCell.__init__(self, *args)
        moose.CaConc(self.soma.path + '/CaPool').tau = 50e-3
	
    def _topology(self):
        self.presyn = 135
        self.level[1].add(self.comp[1])
        for ii  in range(2, 120, 13):
            self.level[2].add(self.comp[ii])
        for ii  in range(3, 121, 13):
            self.level[3].add(self.comp[ii])
            self.level[3].add(self.comp[ii+1])
            self.level[3].add(self.comp[ii+2])
        for ii  in range(6, 124, 13):
            for kk in range(0,9):
                self.level[4].add(self.comp[ii+kk])
        for ii  in range(132, 138):
            self.level[0].add(self.comp[ii])
            
    
    def _setup_passive(self):
        for comp in self.comp[1:]:
            comp.Em = -70e-3
	    comp.initVm = -70e-3

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
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
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")


    @classmethod
    def test_single_cell(cls):
        """Simulates a single thalamocortical relay cell
        and plots the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
        mycell = TCR(TCR.prototype, sim.model.path + "/TCR")
        print 'Created cell:', mycell.path
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_TCR', 'Vm', sim.data)
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
    TCR.test_single_cell()


# 
# tcr.py ends here
