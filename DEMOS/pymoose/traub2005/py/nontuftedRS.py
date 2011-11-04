# nontuftedRS.py --- 
# 
# Filename: nontuftedRS.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Oct 16 11:34:27 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:04:14 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 36
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


class NontuftedRS(TraubCell):
    chan_params = {
        'ENa': 50e-3,
        'EK': -95e-3,
        'EAR': -35e-3,
        'ECa': 125e-3,
        'EGABA': -75e-3, # Sanchez-Vives et al. 1997 
        'TauCa': 20e-3,
        'X_AR': 0.25
        }
    num_comp = 50
    presyn = 48
    level = None
    depth = {
        1: 2200 * 1e-6,
        2: 2245 * 1e-6,
        3: 2290 * 1e-6,
        4: 2335 * 1e-6,
        5: 2175 * 1e-6,
        6: 2125 * 1e-6,
        7: 2075 * 1e-6,
        8: 2025 * 1e-6,
        9: 1975 * 1e-6,
        10: 1925 * 1e-6,
        11: 1875 * 1e-6,
        12: 1825 * 1e-6,
        13: 1775 * 1e-6,
        14: 1725 * 1e-6
        }
    proto_file = 'NontuftedRS.p'
    prototype = TraubCell.read_proto(proto_file, 'NontuftedRS', level_dict=level, depth_dict=depth, params=chan_params)
    ca_dep_chans = ['KAHP_DP', 'KC']
    def __init__(self, *args):
        TraubCell.__init__(self, *args)
        moose.CaConc(self.soma.path + '/CaPool').tau = 100e-3
	
    def _topology(self):
        raise Exception, 'Deprecated'
    
    def _setup_passive(self):
        raise Exception, 'Deprecated'

    def _setup_channels(self):
        """Set up connections between compartment and channels, and Ca pool"""
        raise Exception, 'Deprecated'

    @classmethod
    def test_single_cell(cls):
        """Simulates a single nontufted regular spiking cell and plots
        the Vm and [Ca2+]"""

        config.LOGGER.info("/**************************************************************************")
        config.LOGGER.info(" *")
        config.LOGGER.info(" * Simulating a single cell: %s" % (cls.__name__))
        config.LOGGER.info(" *")
        config.LOGGER.info(" **************************************************************************/")

        sim = Simulation(cls.__name__)
        mycell = NontuftedRS(NontuftedRS.prototype, sim.model.path + "/NontuftedRS")
        print 'Created cell:', mycell.path
        vm_table = mycell.comp[NontuftedRS.presyn].insertRecorder('Vm_nontuftRS', 'Vm', sim.data)
        pulsegen = mycell.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=50e-3, firstWidth=50e-3)
        sim.schedule()
        if mycell.has_cycle():
            print "WARNING!! CYCLE PRESENT IN CICRUIT."
        t1 = datetime.now()
        sim.run(200e-3)
        t2 = datetime.now()
        delta = t2 - t1
        print 'simulation time: ', delta.seconds + 1e-6 * delta.microseconds
        if config.has_pylab:
            mus_vm = config.pylab.array(vm_table) * 1e3
            mus_t = linspace(0, sim.simtime * 1e3, len(mus_vm))
            try:
                nrn_vm = config.pylab.loadtxt('../nrn/mydata/Vm_nontuftRS.plot')
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
    NontuftedRS.test_single_cell()






# 
# nontuftedRS.py ends here
