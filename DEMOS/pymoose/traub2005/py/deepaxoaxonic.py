# deepaxoaxonic.py --- 
# 
# Filename: deepaxoaxonic.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 11:11:39 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:18:59 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 58
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


class DeepAxoaxonic(TraubCell):
    chan_params = {
        'ENa': 50e-3,
        'EK': -100e-3,
        'ECa': 125e-3,
        'EAR': -40e-3,
        'EGABA': -75e-3,
        'X_AR': 0.0,
        'TauCa': 20e-3
        }
    ca_dep_chans = ['KC_FAST']
    num_comp = 59
    presyn = 59
    # level maps level number to the set of compartments belonging to it
    # level = TraubCell.readlevels("DeepAxoaxonic.levels")
    # depth stores a map between level number and the depth of the compartments.
    depth = None
    
    proto_file = "DeepAxoaxonic.p"
    prototype = TraubCell.read_proto(proto_file, "DeepAxoaxonic", chan_params)
    ca_dep_chans = ['KC_FAST']
    def __init__(self, *args):
        TraubCell.__init__(self, *args)
        moose.CaConc(self.soma.path + '/CaPool').tau = 50e-3
	
    def _topology(self):
        raise Exception, 'Deprecated'

    
    def _setup_passive(self):
        raise Exception, 'Deprecated'

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
        raise Exception, 'Deprecated'

    @classmethod
    def test_single_cell(cls):
        """Simulates a single deep axoaxonic cell and plots the Vm and [Ca2+]"""
        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
        mycell = DeepAxoaxonic(DeepAxoaxonic.prototype, sim.model.path + "/DeepAxoaxonic")
        config.LOGGER.info('Created cell: %s' % (mycell.path))
        vm_table = mycell.comp[mycell.presyn].insertRecorder('Vm_deepaxax', 'Vm', sim.data)
        ca_conc_path = mycell.soma.path + '/CaPool'
        ca_table = None
        if config.context.exists(ca_conc_path):
            ca_conc = moose.CaConc(ca_conc_path)
            ca_table = moose.Table('Ca_deepaxax', sim.data)
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
                nrn_vm = config.pylab.loadtxt('../nrn/mydata/Vm_deepaxax.plot')
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
    DeepAxoaxonic.test_single_cell()





# 
# deepaxoaxonic.py ends here
