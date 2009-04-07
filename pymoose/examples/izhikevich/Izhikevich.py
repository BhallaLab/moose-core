# Izhikevich.py --- 
# 
# Filename: Izhikevich.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr  6 15:43:16 2009 (+0530)
# Version: 
# Last-Updated: Tue Apr  7 16:31:34 2009 (+0530)
#           By: subhasis ray
#     Update #: 113
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
import sys
sys.path.append('/usr/lib/python2.5/site-packages')
sys.path.append('/home/subha/src/moose/pymoose')

import moose

class TonicSpikingNrn(moose.IzhikevichNrn):
    """Tonic spiking neuron using Izhikevich model."""
    def __init__(self, *args):
        moose.IzhikevichNrn.__init__(self, *args)
        # using physiological unit
        self.alpha = 0.04
        self.beta = 5.0
        self.gamma = 140.0
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 6.0
        self.Vmax = 30.0
        self.initVm = -70.0
        self.initU = -20

    def setup(self, dt=1.0):
        self.vm_table = moose.Table("/VmTable")
        self.vm_table.stepMode = 3
        self.u_table =  moose.Table("/UTable")
        self.u_table.stepMode = 3
        self.connect("Vm", self.vm_table, "inputRequest")
        self.connect("u", self.u_table, "inputRequest")
        pulseGen = moose.PulseGen("/pulseGen")
        pulseGen.baseLevel = 14.0
        pulseGen.firstLevel = 14.0
        pulseGen.firstDelay = 0.0
        pulseGen.firstWidth = 1e6
        pulseGen.connect("outputSrc", self, "injectDest")
        self.getContext().setClock(0, dt, 0)
        self.getContext().setClock(1, dt, 1)
        self.getContext().useClock(0, '/pulseGen')
        self.getContext().useClock(1, '/VmTable,/UTable,' + self.path)

    def run(self, duration):
        self.getContext().reset()
        self.getContext().step(duration)

    def dump_data(self):
        self.vm_table.dumpFile("tonic_spiking_vm.plot")
        self.u_table.dumpFile("tonic_spiking_u.plot")
        return self.vm_table

    def fullrun(self, duration=1000, dt=1.0):
        self.setup(dt)
        self.run(duration)
        return self.dump_data()

import pylab

def testTonicSpikingNrn(duration=1000, dt=1.0):
    ts = TonicSpikingNrn("/tonicSpiking")
    vm_ts = pylab.array(ts.fullrun(duration, dt))
    steps = int(duration/dt)
    vm = pylab.zeros(steps + 1)
    a = 0.02
    b = 0.2
    c = -65.0
    d = 6.0
    vm[0] = -70.0
    inject = 14.0
    u = -20.0
    vmax = 30.0
    for i in range(steps):
        if vm[i] >= vmax:
            vm[i+1] = c
            u = u + d
        else:
            vm[i+1] = vm[i] * (1.0 + dt * (0.04 * vm[i] + 5.0)) + dt * (140.0 - u + inject)
            u = u + dt * (a * (b * vm[i+1] - u))
        print vm[i+1], u
    pylab.savetxt("tonic_spike_moose.plot", vm_ts)
    pylab.savetxt("tonic_spike_numpy.plot", vm)
    numpy_line = pylab.plot(vm, 'b')
    moose_line = pylab.plot(vm_ts[1:], 'r')
    pylab.legend((numpy_line, moose_line), ('numpy', 'moose'))
    pylab.show()
                

if __name__ == "__main__":
    testTonicSpikingNrn()
    

# 
# Izhikevich.py ends here
