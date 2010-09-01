# Izhikevich.py --- 
# 
# Filename: Izhikevich.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri May 28 14:42:33 2010 (+0530)
# Version: 
# Last-Updated: Wed Aug  4 20:34:46 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 985
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# threhold variablity to be checked.
# Bistability not working.
# DAP working with increased parameter value 'a'
# inhibition induced spiking kind of working but not matching with the paper figure
# inhibition induced bursting kind of working but not matching with the paper figure
# Accommodation cannot work with the current implementation: because the equation for u is not what is mentioned in the paper
# it is: u = u + tau*a*(b*(V+65)); [It is nowhere in the paper and you face it only if you look at the matlab code for figure 1].
# It is not possible to tune a, b, c, d in any way to produce this from: u = u + tau*a*(b*V - u) 
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

from numpy import *

import moose

class IzhikevichDemo:
    """Class to setup and simulate the various kind of neuronal behaviour using Izhikevich model.
    
    Fields:
    """    
    # Paramteres for different kinds of behaviour described by Izhikevich
    # (1. IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003
    # and 2. IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 15, NO. 5, SEPTEMBER
    # 2004)
    # Modified and enhanced using: http://www.izhikevich.org/publications/figure1.m
    # The entries in the tuple are as follows:
    # fig. no. in paper (2), parameter a, parameter b, parameter c (reset value of v in mV), parameter d (after-spike reset value of u), injection current I (mA), initial value of Vm, duration of simulation (ms)
    # 
    # They are all in whatever unit they were in the paper. Just before use we convert them to SI.
    parameters = {
        "tonic_spiking":    ['A', 0.02  ,    0.2  ,   -65.0,     6.0  ,      14.0,      -70.0,  100.0], # Fig. 1.A
        "phasic_spiking":   ['B', 0.02  ,    0.25 ,   -65.0,     6.0  ,      0.5,       -64.0,  200.0], # Fig. 1.B
        "tonic_bursting":   ['C', 0.02  ,    0.2  ,   -50.0,     2.0  ,      15.0,      -70.0,  220.0], # Fig. 1.C
        "phasic_bursting":  ['D', 0.02  ,    0.25 ,   -55.0,     0.05 ,      0.6,       -64.0,  200.0], # Fig. 1.D
        "mixed_mode":       ['E', 0.02  ,    0.2  ,   -55.0,     4.0   ,     10.0,      -70.0,  160.0], # Fig. 1.E
        "spike_freq_adapt": ['F', 0.01  ,    0.2  ,   -65.0,     8.0   ,     30.0,      -70.0,  85.0 ],  # Fig. 1.F # spike frequency adaptation
        "Class_1":          ['G', 0.02  ,    -0.1 ,   -55.0,     6.0   ,     0,         -60.0,  300.0], # Fig. 1.G # Spikining Frequency increases with input strength
        "Class_2":          ['H', 0.2   ,    0.26 ,   -65.0,     0.0   ,     0,         -64.0,  300.0], # Fig. 1.H # Produces high frequency spikes  
        "spike_latency":    ['I', 0.02  ,    0.2  ,   -65.0,     6.0   ,     7.0,       -70.0,  100.0], # Fig. 1.I
        "subthresh_osc":    ['J', 0.05  ,    0.26 ,   -60.0,     0.0   ,     0,         -62.0,  200.0], # Fig. 1.J # subthreshold oscillations
        "resonator":        ['K', 0.1   ,    0.26 ,   -60.0,     -1.0  ,     0,         -62.0,  400.0], # Fig. 1.K 
        "integrator":       ['L', 0.02  ,    -0.1 ,   -55.0,     6.0   ,     0,         -60.0,  100.0], # Fig. 1.L 
        "rebound_spike":    ['M', 0.03  ,    0.25 ,   -60.0,     4.0   ,     -15,       -64.0,  200.0], # Fig. 1.M 
        "rebound_burst":    ['N', 0.03  ,    0.25 ,   -52.0,     0.0   ,     -15,       -64.0,  200.0], # Fig. 1.N 
        "thresh_var":       ['O', 0.03  ,    0.25 ,   -60.0,     4.0   ,     0,         -64.0,  100.0], # Fig. 1.O # threshold variability
        "bistable":         ['P', 0.1   ,    0.26  ,  -60.0,     0.0   ,     1.24,      -61.0,  300.0], # Fig. 1.P 
        "DAP":              ['Q', 1.15   ,    0.2  ,   -60.0,     -21.0 ,     20,        -70.0,  50.0], # Fig. 1.Q # Depolarizing after-potential - a had to be increased in order to reproduce the figure
        "accommodation":    ['R', 0.02  ,    1.0  ,   -55.0,     4.0   ,     0,         -65.0,  400.0], # Fig. 1.R 
        "iispike":          ['S', -0.02 ,    -1.0 ,   -60.0,     8.0   ,     75.0,      -63.8,  350.0], # Fig. 1.S # inhibition-induced spiking
        "iiburst":          ['T', -0.026,    -1.0 ,   -45.0,     0.0   ,     75.0,      -63.8,  350.0]  # Fig. 1.T # inhibition-induced bursting
    }

    documentation = {
        "tonic_spiking":        """
Neuron is normally silent but spikes when stimulated with a current injection.""",

        "phasic_spiking":       """
Neuron fires a single spike only at the start of a current pulse.""",

        "tonic_bursting":       """
Neuron is normally silent but produces bursts of spikes when
stimulated with current injection.""",

        "phasic_bursting":      """
Neuron is normally silent but produces a burst of spikes at the
beginning of an input current pulse.""",

        "mixed_mode":           """
Neuron fires a burst at the beginning of input current pulse, but then
switches to tonic spiking.""",

        "spike_freq_adapt":     """
Neuron fires spikes when a current injection is applied, but at a
gradually reducing rate.""",

        "Class_1":              """
Neuron fires low frequency spikes with weak input current injection.""",

        "Class_2":              """
Neuron fires high frequency (40-200 Hz) spikes when stimulated with
current injection.""",

        "spike_latency":        """
The spike starts after a delay from the onset of current
injection. The delay is dependent on strength of input.""",

        "subthresh_osc":        """
Even at subthreshold inputs a neuron exhibits oscillatory membrane potential.""",

        "resonator":            """
Neuron fires spike only when an input pulsetrain of a frequency
 similar to that of the neuron's subthreshold oscillatory frequency is
 applied.""",

        "integrator":           """
The chances of the neuron firing increases with increase in the frequency 
of input pulse train.""",

        "rebound_spike":        """
When the neuron is released from an inhibitory input, it fires a spike.""",

        "rebound_burst":        """
When the neuron is released from an inhibitory input, it fires a burst
 of action potentials.""",

        "thresh_var":           """
Depending on the previous input, the firing threshold of a neuron may
change.  In this example, the first input pulse does not produce
spike, but when the same input is applied after an inhibitory input,
it fires.""",

        "bistable":             """
These neurons switch between two stable modes (resting and tonic spiking). 
The switch happens via an excitatory or inhibitory input.""",

        "DAP":                  """
After firing a spike, the membrane potential shows a prolonged depolarized 
after-potential.""",

        "accommodation":        """
These neurons do not respond to slowly rising input, but a sharp increase 
in input may cause firing.""",

        "iispike":              """
These neurons fire in response to inhibitory input.""",

        "iiburst":              """
These neurons show bursting in response to inhibitory input."""
        }
    
    def __init__(self):
        """Initialize the object."""
        self.neurons = {}
        self.Vm_tables = {}
        self.u_tables = {}
        self.inject_tables = {}
        self.inputs = {}
        self.context = moose.PyMooseBase.getContext()
        self.simtime = 100e-3
        self.dt = 0.25e-3
        self.steps = int(self.simtime/self.dt)
        self.context.setClock(0, self.dt)
        self.context.setClock(1, self.dt)
        self.context.setClock(2, self.dt)
        self.neuron = None

    def setup(self, key):
        neuron = self._get_neuron(key)
        pulsegen = self._make_pulse_input(key)
        if pulsegen is None:
            print key, 'Not implemented.'
            
    def simulate(self, key):
        if key == 'accommodation':
            raise NotImplementedError('Not Implemented', 'Equation for u for the accommodating neuron is:\n u\' = a * b * (V + 65)\n Which is different from the regular equation and cannot be obtained from the latter by any choice of a and b.')
            return
        self.setup(key)
        return self.run(key)

    def run(self, key):
        try:
            Vm = self.Vm_tables[key]
            u = self.u_tables[key]
        except KeyError, e:
            Vm = moose.Table(key + '_Vm')
            Vm.stepMode = 3
            Vm.connect('inputRequest', self.neurons[key], 'Vm')
            utable = moose.Table(key + '_u')
            utable.stepMode = 3
            utable.connect('inputRequest', self.neurons[key], 'u')
            self.Vm_tables[key] = Vm
            self.u_tables[key] = utable
        try:
            Im = self.inject_tables[key]
        except KeyError, e:
            Im = moose.Table(key + '_inject') # May be different for non-pulsegen sources.
            Im.stepMode = 3
            Im.connect('inputRequest', self._get_neuron(key), 'Im')
            self.inject_tables[key] = Im
        self.simtime = IzhikevichDemo.parameters[key][7] * 1e-3
        self.context.reset()
        self.context.step(self.simtime)
        time = linspace(0, IzhikevichDemo.parameters[key][7], len(Vm))
        # DEBUG
        nrn = self._get_neuron(key)
        print 'a = %g, b = %g, c = %g, d = %g, initVm = %g, initU = %g' % (nrn.a,nrn.b, nrn.c, nrn.d, nrn.initVm, nrn.initU)
        #! DEBUG
        return (time, Vm, Im)


    def _get_neuron(self, key):
        try:
            neuron = self.neurons[key]
            return neuron
        except KeyError, e:
            pass
        try:
            params = IzhikevichDemo.parameters[key]
        except KeyError, e:
            print ' %s : Invalid neuron type. The valid types are:' % (key)
            for key in IzhikevichDemo.parameters:
                print key
            raise
        neuron = moose.IzhikevichNrn(key)
        if key == 'integrator' or key == 'Class_1': # Integrator has different constants
            neuron.beta = 4.1e3
            neuron.gamma = 108.0
        self.neuron = neuron
        neuron.a = params[1] * 1e3 # ms^-1 -> s^-1
        neuron.b = params[2] * 1e3 # ms^-1 -> s^-1
        neuron.c = params[3] * 1e-3 # mV -> V
        neuron.d = params[4]  # d is in mV/ms = V/s
        neuron.initVm = params[6] * 1e-3 # mV -> V
        neuron.Vmax = 0.03 # mV -> V
        if key != 'accommodation':
            neuron.initU = neuron.initVm * neuron.b
        else:
            neuron.initU = -16.0 # u is in mV/ms = V/s
        self.neurons[key] = neuron
        return neuron

    def _make_pulse_input(self, key):
        """This is for creating a pulse generator for use as a current
        source for all cases except Class_1, Class_2, resonator,
        integrator, thresh_var and accommodation."""
        try:
            return self.inputs[key]
        except KeyError:
            pass # continue to the reset of the function
        baseLevel = 0.0
        firstWidth = 1e6
        firstDelay = 0.0
        firstLevel = IzhikevichDemo.parameters[key][5] * 1e-9
        secondDelay = 1e6
        secondWidth = 0.0
        secondLevel = 0.0
        if key == 'tonic_spiking':
            firstDelay = 10e-3
        elif key == 'phasic_spiking':
            firstDelay = 20e-3
        elif key == 'tonic_bursting':
            firstDelay = 22e-3
        elif key == 'phasic_bursting':
            firstDelay = 20e-3
        elif key == 'mixed_mode':
            firstDelay = 16e-3
        elif key == 'spike_freq_adapt':
            firstDelay = 8.5e-3
        elif key == 'spike_latency':
            firstDelay = 10e-3
            firstWidth = 3e-3
        elif key == 'subthresh_osc':
            firstDelay = 20e-3
            firstWidth = 5e-3
            firstLevel = 2e-9
        elif key == 'rebound_spike':
            firstDelay = 20e-3
            firstWidth = 5e-3
        elif key == 'rebound_burst':
            firstDelay = 20e-3
            firstWidth = 5e-3
        elif key == 'bistable':
            input_table = self._make_bistable_input()
            self.inputs[key] = input_table
            return input_table
        elif key == 'DAP':
            firstDelay = 9e-3
            firstWidth = 2e-3
        elif (key == 'iispike') or (key == 'iiburst'):
            baseLevel = 80e-9
            firstDelay = 50e-3
            firstWidth = 200e-3
            fisrtLevel = 75e-9
        elif key == 'Class_1':
            input_table = self._make_Class_1_input()
            self.inputs[key] = input_table
            return input_table
        elif key == 'Class_2':
            input_table = self._make_Class_2_input()
            self.inputs[key] = input_table
            return input_table
        elif key == 'resonator':
            input_table = self._make_resonator_input()
            self.inputs[key] = input_table
            return input_table
        elif key == 'integrator':
            input_table = self._make_integrator_input()
            self.inputs[key] = input_table
            return input_table
        elif key == 'accommodation':
            input_table = self._make_accommodation_input()
            self.inputs[key] = input_table
            return input_table            
        elif key == 'thresh_var':
            input_table = self._make_thresh_var_input()
            self.inputs[key] = input_table
            return input_table                        
        else:
            print key, ': Stimulus is not based on pulse generator.'
            raise
        pulsegen = self._make_pulsegen(key, 
                                      firstLevel,
                                      firstDelay,
                                      firstWidth,
                                      secondLevel,
                                      secondDelay,
                                      secondWidth, baseLevel)
        self.inputs[key] = pulsegen
        return pulsegen
                                               

    def _make_pulsegen(self, key, firstLevel, firstDelay, firstWidth=1e6, secondLevel=0, secondDelay=1e6, secondWidth=0, baseLevel=0):
        pulsegen = moose.PulseGen(key + '_input')
        pulsegen.firstLevel = firstLevel
        pulsegen.firstDelay = firstDelay
        pulsegen.firstWidth = firstWidth
        pulsegen.secondLevel = secondLevel
        pulsegen.secondDelay = secondDelay
        pulsegen.secondWidth = secondWidth
        pulsegen.baseLevel = baseLevel
        pulsegen.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return pulsegen    
        
    def _make_Class_1_input(self):
        input_table = moose.Table('Class_1_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin = 30e-3 # The ramp starts at 30 ms
        input_table.xmax = IzhikevichDemo.parameters['Class_1'][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        input_table[0] = 0.0
        for i in range(1, len(input_table)):
            # matlab code: if (t>T1) I=(0.075*(t-T1)); else I=0;
            input_table[i] = (0.075 * i * self.dt * 1e3) * 1e-9
        input_table.connect('outputSrc', self._get_neuron('Class_1'), 'injectDest')
        return input_table

    def _make_Class_2_input(self):
        key = 'Class_2'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin = 30e-3 # The ramp starts at 30 ms
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        input_table[0] = -0.5e-9
        for i in range(1, len(input_table)):
            # The matlab code is: if (t>T1) I=-0.5+(0.015*(t-T1)); else I=-0.5
            input_table[i] = (-0.5 + 0.015 * i * self.dt * 1e3) * 1e-9 # convert dt from s to ms, and convert total current from nA to A.
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table

    def _make_bistable_input(self):
        key = 'bistable'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin =  0
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        t1 = IzhikevichDemo.parameters[key][7] * 1e-3/8
        t2 = 216e-3
        t = 0.0
        for ii in range(len(input_table)):
            if (t > t1 and t < t1 + 5e-3) or (t > t2 and t < t2 + 5e-3):
                input_table[ii] = 1.24e-9
            else:
                input_table[ii] = 0.24e-9
            t = t + self.dt
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table

    def _make_resonator_input(self):
        key = 'resonator'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin =  0
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        t1 = IzhikevichDemo.parameters[key][7] * 1e-3/10
        t2 = t1 + 20e-3
        t3 = 0.7 * IzhikevichDemo.parameters[key][7] * 1e-3
        t4 = t3 + 40e-3
        t = 0.0
        for ii in range(len(input_table)):
            if (t > t1 and t < t1 + 4e-3) or (t > t2 and t < t2 + 4e-3) or (t > t3 and t < t3 + 4e-3) or (t > t4 and t < t4 + 4e-3):
                input_table[ii] = 0.65e-9
            else:
                input_table[ii] = 0.0
            t = t + self.dt
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table
        
    def _make_integrator_input(self):
        key = 'integrator'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin =  0
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        t1 = IzhikevichDemo.parameters[key][7] * 1e-3/11
        t2 = t1 + 5e-3
        t3 = 0.7 * IzhikevichDemo.parameters[key][7] * 1e-3
        t4 = t3 + 10e-3
        t = 0.0
        for ii in range(len(input_table)):
            if (t > t1 and t < t1 + 2e-3) or (t > t2 and t < t2 + 2e-3) or (t > t3 and t < t3 + 2e-3) or (t > t4 and t < t4 + 2e-3):
                input_table[ii] = 9e-9
            else:
                input_table[ii] = 0.0
            t = t + self.dt
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table
        
    def _make_accommodation_input(self):
        key = 'accommodation'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin =  0
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        t = 0.0
        for ii in range(len(input_table)):
            if t < 200e-3:
                input_table[ii] = t * 1e-9/25
            elif t < 300e-3:
                input_table[ii] = 0.0
            elif t < 312.5e-3:
                input_table[ii] = 4e-9 * (t-300e-3)/12.5
            else:
                input_table[ii] = 0.0
            t = t + self.dt
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table
        
    def _make_thresh_var_input(self):
        key = 'thresh_var'
        input_table = moose.Table(key + '_input')
        input_table.stepMode = 1 # Table acts as a function generator
        input_table.stepSize = self.dt
        input_table.xmin =  0
        input_table.xmax = IzhikevichDemo.parameters[key][7] * 1e-3
        input_table.xdivs = int(ceil((input_table.xmax - input_table.xmin) / input_table.stepSize))
        t = 0.0
        for ii in range(len(input_table)):
            if (t > 10e-3 and t < 15e-3) or (t > 80e-3 and t < 85e-3):
                input_table[ii] = 1e-9
            elif t > 70e-3 and t < 75e-3:
                input_table[ii] = -6e-9
            else:
                input_table[ii] = 0.0
            t = t + self.dt
        input_table.connect('outputSrc', self._get_neuron(key), 'injectDest')
        return input_table

    def getEquation(self, key):
        params = IzhikevichDemo.parameters[key]
        equationText = "<i>v' = 0.04v^2 + 5v + 140 - u + I</i><br><i>u' = a(bv - u)</i><p>If <i>v >= 30 mV, v = c</i> and <i>u = u + d</i><br>where <i>a = %g</i>, <i>b = %g</i>, <i>c = %g</i> and <i>d = %g</i>."  % (params[1], params[2], params[3], params[4])
        return equationText
        
import sys
try:
    from pylab import *
    if __name__ == '__main__':
        key = 'tonic_spiking'
        if len(sys.argv) > 1:
            key = sys.argv[1]
        demo = IzhikevichDemo()
        (time, Vm, Im) = demo.simulate(key)
        title(IzhikevichDemo.parameters[key][0] + '. ' + key)
        subplot(2,1,1)
        plot(time, array(Vm))
        subplot(2,1,2)
        plot(time, array(Im))
        show()
        # data.dumpFile(data.name + '.plot')
        # demo.inject_table.dumpFile(demo.inject_table.name + '.plot')
        print 'Finished simulation.'
except ImportError:
    print 'Matplotlib not installed.'

# 
# Izhikevich.py ends here
