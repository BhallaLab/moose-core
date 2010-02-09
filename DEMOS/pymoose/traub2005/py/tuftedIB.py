# tuftedIB.py --- 
# 
# Filename: tuftedIB.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 11:44:48 2009 (+0530)
# Version: 
# Last-Updated: Tue Feb  9 14:29:23 2010 (+0100)
#           By: Subhasis Ray
#     Update #: 41
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
import trbutil
import moose
from cell import *
from capool import CaPool


class TuftedIB(TraubCell):
    prototype = TraubCell.read_proto("TuftedIB.p", "TuftedIB")
    ca_dep_chans = ['KAHP','KAHP_SLOWER', 'KAHP_DP', 'KC', 'KC_FAST']
    def __init__(self, *args):
	TraubCell.__init__(self, *args)
	
    def _topology(self):
        self.presyn = 60
    
    def _setup_passive(self):
        for comp in self.comp[1:]:
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
		    ca_pool.tau = 1e-3/0.075
		else:
		    obj_class = obj.className
		    if obj_class == 'HHChannel':
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
	    if ca_pool:
		for channel in ca_chans:
		    channel.connect('IkSrc', ca_pool, 'current')
		for channel in ca_dep_chans:
		    channel.useConcentration = 1
		    ca_pool.connect("concSrc", channel, "concen")

	obj = moose.CaConc(self.soma.path + '/CaPool')
        obj.tau = 100e-3
	# Special case: individually specified beta_cad's in level  2
	moose.CaConc(self.comp[2].path + '/CaPool').tau  =   1e-3/0.02
        moose.CaConc(self.comp[ 3].path + '/CaPool' ).tau = 1e-3 /  0.075
        moose.CaConc(self.comp[ 4].path + '/CaPool' ).tau = 1e-3 /  0.075
        moose.CaConc(self.comp[ 5].path + '/CaPool' ).tau = 1e-3 /  0.02
        moose.CaConc(self.comp[ 6].path + '/CaPool' ).tau = 1e-3 /  0.02
        moose.CaConc(self.comp[ 7].path + '/CaPool' ).tau = 1e-3 /  0.075
        moose.CaConc(self.comp[ 8].path + '/CaPool' ).tau = 1e-3 /  0.075
        moose.CaConc(self.comp[ 9].path + '/CaPool' ).tau = 1e-3 /  0.075
        moose.CaConc(self.comp[ 10].path + '/CaPool' ).tau = 1e-3 / 0.075
        moose.CaConc(self.comp[ 11].path + '/CaPool' ).tau = 1e-3 / 0.075
        moose.CaConc(self.comp[ 12].path + '/CaPool' ).tau = 1e-3 / 0.075


    @classmethod
    def test_single_cell(cls):
        """Simulates a single tufted intrinsically bursting cell and
        plots the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
        mycell = TuftedIB(TuftedIB.prototype, sim.model.path + "/TuftedIB")
        print 'Created cell:', mycell.path
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_tuftIB', 'Vm', sim.data)
        ca_table = mycell.soma.insertCaRecorder('CaPool', sim.data)
        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=0.3e-9, firstDelay=0.0, firstWidth=50e-3)
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
        mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))
        mus_ca = pylab.array(ca_table)
        nrn_vm = trbutil.read_nrn_data('Vm_tuftIB.plot', 'test_tuftIB.hoc')
        nrn_ca = trbutil.read_nrn_data('Ca_tuftIB.plot', 'test_tuftIB.hoc')
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
    TuftedIB.test_single_cell()



# 
# tuftedIB.py ends here
