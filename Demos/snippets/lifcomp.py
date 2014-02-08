# lifcomp.py --- 
# 
# Filename: lifcomp.py
# Description: Leaky Integrate and Fire using regular neuronal compartment
# Author: subha
# Maintainer: 
# Created: Fri Feb  7 16:26:05 2014 (+0530)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
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

"""This is an example of how you can create a Leaky Integrate and Fire
compartment using regular compartment and Func to check for thresold
crossing and resetting the Vm."""

import moose
from moose import utils
from pylab import *

class LIFComp(moose.Compartment):
    """Leaky integrate and fire neuron using regular compartments,
    spikegen and Func."""
    def __init__(self, *args):
        moose.Compartment.__init__(self, *args)
        self.spikegen = moose.SpikeGen('%s/spike' % (self.path))
        self.spikegen.edgeTriggered = 1 # This ensures that spike is generated only on leading edge.
        self.dynamics = moose.Func('%s/dynamics' % (self.path))
        self.dynamics.expr = 'x > Vthreshold? Vreset: x'
        moose.connect(self, 'VmOut', self.dynamics, 'xIn')
        moose.connect(self.dynamics, 'valueOut', self, 'setVm')
        moose.connect(self, 'VmOut', self.spikegen, 'Vm')

    @property
    def Vreset(self):
        return self.dynamics.var['Vreset']

    @Vreset.setter
    def Vreset(self, value):
        self.dynamics.var['Vreset'] = value

    @property
    def Vthreshold(self):
        return self.dynamics.var['Vthreshold']

    @Vthreshold.setter
    def Vthreshold(self, value):
        self.dynamics.var['Vthreshold'] = value
        self.spikegen.threshold = value

def setup_two_cells():
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    a1 = LIFComp('/model/a1')
    a2 = LIFComp('/model/a2')
    moose.connect(a1, 'raxial', a2, 'axial')
    b1 = LIFComp('/model/b1')
    b2 = LIFComp('/model/b2')
    moose.connect(b1, 'raxial', b2, 'axial')
    a1.Vthreshold = -55e-3
    a1.Vreset = -70e-3
    a2.Vthreshold = -55e-3
    a2.Vreset = -70e-3
    b1.Vthreshold = -55e-3
    b1.Vreset = -70e-3
    b2.Vthreshold = -55e-3
    b2.Vreset = -70e-3
    ## Adding a SynChan gives a core dump
    syn = moose.SynChan('%s/syn' % (b2.path))
    syn.synapse.num = 1
    moose.connect(b2, 'channel', syn, 'channel')
    ## below gives a core dump on running
    #moose.connect(a1.spikegen, 'spikeOut',
    #              syn.synapse.vec, 'addSpike')
    stim = moose.PulseGen('stim')
    stim.delay[0] = 100e-3
    stim.width[0] = 10e-3
    stim.level[0] = 1e-9
    ## below gives error: NameError: check field names and type compatibility.
    #moose.connect(stim, 'output', a1, 'injectMsg')
    tables = []
    for c in moose.wildcardFind('/##[ISA=Compartment]'):
        c.Rm = 1e6
        c.Ra = 1e4
        c.Cm = 1e-9
        c.inject = 5.1e-9
        tables.append( utils.setupTable("vmTable",a1,'Vm') )
    #syn.synapse[0].delay = 1e-3
    #syn.Gk = 1e-9
    return tables

if __name__ == '__main__':
    tables = setup_two_cells()
    utils.setDefaultDt()
    utils.assignDefaultTicks(solver='ee')
    utils.stepRun(1.0, 100e-3)    
    plot(tables[0].vector)
    show()
# 
# lifcomp.py ends here
