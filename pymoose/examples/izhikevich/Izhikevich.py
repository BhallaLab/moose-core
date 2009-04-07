# Izhikevich.py --- 
# 
# Filename: Izhikevich.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr  6 15:43:16 2009 (+0530)
# Version: 
# Last-Updated: Tue Apr  7 20:13:24 2009 (+0530)
#           By: subhasis ray
#     Update #: 233
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

from math import *

import sys
sys.path.append('/usr/lib/python2.5/site-packages')
sys.path.append('/home/subha/src/moose/pymoose')

import pylab

import moose

pars={"tonic_spiking":    [0.02  ,    0.2  ,   -65,     6  ,      14 ],
      "phasic_spiking":   [0.02  ,    0.25 ,   -65,     6  ,      0.5],
      "tonic_bursting":   [0.02  ,    0.2  ,   -50,     2  ,      15 ],
      "phasic_bursting":  [0.02  ,    0.25 ,   -55,     0.05,     0.6],
      "mixed_mode":       [0.02  ,    0.2  ,   -55,     4   ,     10 ],
      "spike_freq_adapt": [0.01  ,    0.2  ,   -65,     8   ,     30 ],# spike frequency adaptation
      "Class_1":          [0.02  ,    -0.1 ,   -55,     6   ,     0  ],
      "Class_2":          [0.2   ,    0.26 ,   -65,     0   ,     0  ],
      "spike_latency":    [0.02  ,    0.2  ,   -65,     6   ,     7  ],
      "subthresh_osc":    [0.05  ,    0.26 ,   -60,     0   ,     0  ],	# subthreshold oscillations
      "resonator":        [0.1   ,    0.26 ,   -60,     -1  ,     0  ],
      "integrator":       [0.02  ,    -0.1 ,   -55,     6   ,     0  ],
      "rebound_spike":    [0.03  ,    0.25 ,   -60,     4   ,     0  ],
      "rebound_burst":    [0.03  ,    0.25 ,   -52,     0   ,     0  ],
      "thresh_var":       [0.03  ,    0.25 ,   -60,     4   ,     0  ],	# threshold variability
      "bistable":         [1     ,    1.5  ,   -60,     0   ,     -65],	# bistability
      "DAP":              [  1   ,    0.2  ,   -60,     -21 ,     0  ],
      "accomodation":     [0.02  ,    1    ,   -55,     4   ,     0  ],
      "iispike":          [-0.02  ,   -1   ,   -60,     8   ,     80 ],	# inhibition-induced spiking
      "iiburst":          [-0.026 ,   -1   ,   -45,     0   ,     80 ]}       # inhibition-induced bursting


class IzhikevichTest(moose.IzhikevichNrn):
    def __init__(self, *args):
        moose.IzhikevichNrn.__init__(self, *args)
        # using physiological unit
        self.alpha = 0.04
        self.beta = 5.0
        self.gamma = 140.0
        self.Vmax = 30.0
        self.initVm = -70.0
        self.initU = -20.0
        self.vm_table = moose.Table("VmTable", self)
        self.vm_table.stepMode = 3
        self.inject_table =  moose.Table("injectTable", self)
        self.inject_table.stepMode = 3
        self.connect("Vm", self.vm_table, "inputRequest")
        # PulseGen may be redundant for some types of behaviour like
        # spontaneous firing. Still - for convenience
        self.pulsegen = moose.PulseGen("pulseGen", self)
        self.pulsegen.baseLevel = 0.0
        self.pulsegen.firstDelay = 10.0
        self.pulsegen.firstWidth = 1e6
        self.pulsegen.connect("outputSrc", self, "injectDest")
        self.pulsegen.connect("output", self.inject_table, "inputRequest")
    
    def schedule(self, dt=1.0):
        """Assigns clocks to the model components."""
        self.getContext().setClock(0, dt, 0)
        self.getContext().setClock(1, dt, 1)
        self.pulsegen.useClock(0)
        self.vm_table.useClock(1)
        self.inject_table.useClock(1)
        self.useClock(1)

    def set_type(self, name):
        """Parameterizes the model according to its type"""
        global pars
        props = pars[name]
        if props is None:
            print name, ": no such neuron type in dictionary. falling back to tonic spiking."
            props = pars["tonic_spiking"]
        self.a = props[0]
        self.b = props[1]
        self.c = props[2]
        self.d = props[3]
        self.pulsegen.firstLevel = props[4]

    def dump_data(self):
        self.vm_table.dumpFile(self.name + "_vm.plot")
        self.inject_table.dumpFile(self.name + "_i.plot")
        return self.vm_table

    def fullrun(self, duration=200, dt=1.0):
        self.schedule(dt)
        run(duration)
        return self.dump_data()

def run(duration):
    """Runs the simulation."""
    moose.PyMooseBase.getContext().reset()
    moose.PyMooseBase.getContext().step(duration)

def run_all(duration=200, dt=1.0):
    """Run all the models and show the plots."""
    global pars
    neurons = []
    data = {}
    for nrn_type in pars.keys():
        nrn = IzhikevichTest(nrn_type)
        nrn.set_type(nrn_type)
        nrn.schedule(dt)
        neurons.append(nrn)
    run(duration)
    row_count = ceil(sqrt(len(neurons)))
    col_count = ceil(len(neurons) / row_count)
    plot_num = 1
    for nrn in neurons:
        print nrn.path
        data = pylab.array(nrn.dump_data())
        data[pylab.isnan(data)] = 0.0
        data[pylab.isinf(data)] = 1e3
        inject = pylab.array(nrn.inject_table)
        pylab.subplot(row_count, col_count, plot_num)
        pylab.plot(data)
        pylab.plot(inject - 120.0)
        pylab.title(nrn.name)
        plot_num = plot_num + 1
#    pylab.legend()
if __name__ == "__main__":
    run_all()
    pylab.show()
    

# 
# Izhikevich.py ends here
