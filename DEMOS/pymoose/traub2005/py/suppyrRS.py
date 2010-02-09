# suppyrrs.py --- 
# 
# Filename: suppyrrs.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Aug  7 13:59:30 2009 (+0530)
# Version: 
# Last-Updated: Tue Feb  9 14:29:27 2010 (+0100)
#           By: Subhasis Ray
#     Update #: 601
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: Superficial Regular Spiking Pyramidal Cells of layer 2/3
# From Traub et all, 2005
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

class SupPyrRS(TraubCell):
    prototype = TraubCell.read_proto("SupPyrRS.p", "SupPyrRS")
    ca_dep_chans = ['KAHP','KAHP_SLOWER', 'KAHP_DP', 'KC', 'KC_FAST']
    def __init__(self, *args):
	TraubCell.__init__(self, *args)
	
    def _topology(self):
        self.presyn = 72
	self.level[1].add(self.comp[1])
	for ii in range(2,14):
	    self.level[2].add(self.comp[ii])
	for ii in range(14, 26):
	    self.level[3].add(self.comp[ii])
	for ii in range(26, 38):
	    self.level[4].add(self.comp[ii])
	self.level[5].add(self.comp[38])
	self.level[6].add(self.comp[39])
	self.level[7].add(self.comp[40])
	self.level[8].add(self.comp[41])
	self.level[8].add(self.comp[42])
	self.level[9].add(self.comp[43])
	self.level[9].add(self.comp[44])
	for ii in range(45, 53):
	    self.level[10].add(self.comp[ii])
	for ii in range(53, 61):
	    self.level[11].add(self.comp[ii])
	for ii in range(61, 69):
	    self.level[12].add(self.comp[ii])
	for ii in range(69, 75):
	    self.level[0].add(self.comp[ii])
    
    def _setup_passive(self):
        for comp in self.comp[1:]:
            comp.Em = -70e-3

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
        for i in range(len(self.level)):
            for comp in self.level[i]:
                ca_pool = None
                ca_dep_chans = []
                ca_chans = []
                for child in comp.children():
                    obj = moose.Neutral(child)
                    if obj.name == 'CaPool':
                        ca_pool = moose.CaConc(child)
                        ca_pool.B = ca_pool.B * 1e3
                        ca_pool.tau = 1e-3/0.05
                    else:
                        obj_class = obj.className
                        if obj_class == "HHChannel":
                            obj = moose.HHChannel(child)
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
                if ca_pool: # Setup connections for CaPool : from CaL, to KAHP and KC
                    for channel in ca_chans:
                        channel.connect('IkSrc', ca_pool, 'current')

                    for channel in ca_dep_chans:
                        channel.useConcentration = 1
                        ca_pool.connect("concSrc", channel, "concen")



        obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 1e-3 / 0.01
        config.LOGGER.debug(obj.path + 'set tau to %g' % (obj.tau))

    @classmethod
    def test_single_cell(cls):
        """Simulates a single superficial pyramidal regula spiking
        cell and plots the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
        mycell = SupPyrRS(SupPyrRS.prototype, sim.model.path + "/SupPyrRS")
        config.LOGGER.info('Created cell: %s' % (mycell.path))
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_suppyrrs', 'Vm', sim.data)
        ca_table = mycell.soma.insertCaRecorder('CaPool', sim.data)

        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=50e-3, firstWidth=50e-3)


        sim.schedule()
        if mycell.has_cycle():
            config.LOGGER.warning("WARNING!! CYCLE PRESENT IN CICRUIT.")
        t1 = datetime.now()
        sim.run(200e-3)
        t2 = datetime.now()
        delta = t2 - t1
        print 'simulation time: ', delta.seconds + 1e-6 * delta.microseconds
        sim.dump_data('data')
        mus_vm = pylab.array(vm_table) * 1e3
        mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))
        mus_ca = pylab.array(ca_table)
        nrn_vm = trbutil.read_nrn_data('Vm_suppyrRS.plot', 'test_suppyrRS.hoc')
        nrn_ca = trbutil.read_nrn_data('Ca_suppyrRS.plot', 'test_suppyrRS.hoc')
        if len(nrn_vm) > 0:
            nrn_t = nrn_vm[:, 0]
            nrn_vm = nrn_vm[:, 1]
            nrn_ca = nrn_ca[:,1]

        trbutil.do_plot(cls.__name__, mus_t, mus_ca, mus_vm, nrn_t, nrn_ca, nrn_vm)
        
        
# test main --
from simulation import Simulation
import pylab
from subprocess import call
if __name__ == "__main__":
    SupPyrRS.test_single_cell()
    

# 
# suppyrrs.py ends here
