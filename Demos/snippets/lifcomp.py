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

simtime = 500e-3 # Total simulation time
stepsize = 100e-3 # Time step for pauses between runs
simdt = 1e-4 # time step for numerical integration
plotdt = 0.25e-3 # time step for plotting

delayMax = 0.1 # Maximum synaptic delay 

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
        """Reset voltage. The cell's membrane potential is set to this value
        after spiking."""
        return self.dynamics.var['Vreset']

    @Vreset.setter
    def Vreset(self, value):
        self.dynamics.var['Vreset'] = value

    @property
    def Vthreshold(self):
        """Threshold voltage. The cell spikes if its membrane potential goes
        above this value."""
        return self.dynamics.var['Vthreshold']

    @Vthreshold.setter
    def Vthreshold(self, value):
        self.dynamics.var['Vthreshold'] = value
        self.spikegen.threshold = value

def setup_two_cells():
    """Create two cells with leaky integrate and fire compartments. The
    first cell is composed of two compartments a1 and a2 and the
    second cell is composed of compartments b1 and b2. Each pair is
    connected via raxial message so that the voltage of one
    compartment influences the other through axial resistance Ra. 

    The compartment a1 of the first neuron is connected to the
    compartment b2 of the second neuron through a synaptic channel.

    """
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
    a2.Vthreshold = -59e-3
    a2.Vreset = -70e-3
    b1.Vthreshold = -58e-3
    b1.Vreset = -70e-3
    b2.Vthreshold = -57e-3
    b2.Vreset = -70e-3
    ## Adding a SynChan gives a core dump
    syn = moose.SynChan('%s/syn' % (b2.path))
    syn.bufferTime = delayMax * 2
    syn.numSynapses = 1
    syn.synapse.delay = delayMax
    moose.connect(b2, 'channel', syn, 'channel')
    m = moose.connect(a1.spikegen, 'spikeOut',
                  syn.synapse.vec, 'addSpike', 'Sparse')
    m.setRandomConnectivity(1.0, 1)
    stim = moose.PulseGen('stim')
    stim.delay[0] = 10e-3
    stim.width[0] = 10e-3
    stim.level[0] = 0.0 # 1e-9
    ## current injection doesn't seem to work.
    moose.connect(stim, 'output', a1, 'injectMsg')
    tables = []
    data = moose.Neutral('/data')    
    for c in moose.wildcardFind('/##[ISA=Compartment]'):
        c.Rm = 1e6
        c.Ra = 1e4
        c.Cm = 1e-9
        c.Em = -65e-3
        c.initVm = c.Em
        # c.inject = 5.1e-9
        tab = moose.Table('%s/%s' % (data.path, c.name))
        moose.connect(tab, 'requestOut', c, 'getVm')
        tables.append( tab )
    syn.synapse[0].delay = 1e-3
    syn.Gk = 1e-9
    return tables

if __name__ == '__main__':
    tables = setup_two_cells()
    utils.setDefaultDt(elecdt=simdt, plotdt2=plotdt)
    utils.assignDefaultTicks(solver='ee')
    moose.reinit()
    utils.stepRun(simtime, stepsize)
    for ii, tab in enumerate(tables):
        subplot(len(tables), 1, ii+1)
        t = np.linspace(0, simtime, len(tab.vector))*1e3
        plot(t, tab.vector*1e3, label=tab.name)
        legend()
    show()
# 
# lifcomp.py ends here
