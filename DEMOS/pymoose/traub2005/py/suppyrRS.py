# suppyrrs.py --- 
# 
# Filename: suppyrrs.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Aug  7 13:59:30 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:12:04 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 605
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
    """Superficial Pyramidal Regula Spiking cell."""
    chan_params = {
        'ENa': 50e-3,
        'EK': -95e-3,
        'ECa': 125e-3,
        'EAR': -35e-3,
        'EGABA': -81e-3,
        'TauCa': 20e-3
        }
    ca_dep_chans = ['KAHP', 'KC']
    num_comp = 74
    presyn = 72
    proto_file = "SupPyrRS.p"
    # level maps level number to the set of compartments belonging to it
    level = None
    # depth stores a map between level number and the depth of the compartments.
    depth = {
        1: 850.0 * 1e-6,
        2: 885.0 * 1e-6,
        3: 920.0 * 1e-6,
        4: 955.0 * 1e-6,
        5: 825.0 * 1e-6,
        6: 775.0 * 1e-6,
        7: 725.0 * 1e-6,
        8: 690.0 * 1e-6,
        9: 655.0 * 1e-6,
        10: 620.0 * 1e-6,
        11: 585.0 * 1e-6,
        12: 550.0 * 1e-6
        }
    prototype = TraubCell.read_proto(proto_file, "SupPyrRS", level_dict=level, depth_dict=depth, params=chan_params)
    
    def __init__(self, *args):
        # start = datetime.now()
        TraubCell.__init__(self, *args)
        soma_ca_pool = moose.CaConc(self.soma.path + '/CaPool')
        soma_ca_pool.tau = 100e-3
        # end = datetime.now()
        # delta = end - start
        # config.BENCHMARK_LOGGER.info('created cell in: %g s' % (delta.days * 86400 + delta.seconds + delta.microseconds * 1e-6))
        

    def _topology(self):
        raise Exception, 'Deprecated'
    
    def _setup_passive(self):
        raise Exception, 'Deprecated'

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
        raise Exception, 'Deprecated'

    @classmethod
    def test_single_cell(cls):
        """Simulates a single superficial pyramidal regular spiking
        cell and plots the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")
        sim = Simulation(cls.__name__)
        mycell = SupPyrRS(SupPyrRS.prototype, sim.model.path + "/SupPyrRS")
        config.LOGGER.info('Created cell: %s' % (mycell.path))
        vm_table = mycell.comp[SupPyrRS.presyn].insertRecorder('Vm_suppyrrs', 'Vm', sim.data)
        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=50e-3, firstWidth=50e-3)

        sim.schedule()
        if mycell.has_cycle():
            config.LOGGER.warning("WARNING!! CYCLE PRESENT IN CICRUIT.")
        t1 = datetime.now()
        sim.run(200e-3)
        t2 = datetime.now()
        delta = t2 - t1
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
    SupPyrRS.test_single_cell()
    

# 
# suppyrrs.py ends here
