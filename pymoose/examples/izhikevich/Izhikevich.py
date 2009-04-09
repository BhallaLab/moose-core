# Izhikevich.py --- 
# 
# Filename: Izhikevich.py
# Description: 
#  This is a PyMOOSE implementation of twenty different kinds 
# of spiking neurons described in "Which Model to Use for Cortical 
# Spiking Neurons?" by Eugene M Izhikevich. IEE Transactions on 
# Neural Networks, VOL 15. No. 5. Sept 2004. pp 1063-1070.
#
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr  6 15:43:16 2009 (+0530)
# Version: 
# Last-Updated: Thu Apr  9 18:33:24 2009 (+0530)
#           By: subhasis ray
#     Update #: 566
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: Initial PyQt version of the GUI. The inputs are not
#       right. So the behaviour is strange for some types. Need to
#       fix.
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

class SimEnv:
    """Global simulation environment variables"""
    context = moose.PyMooseBase.getContext() # global context object
    dt = 1.0 # integration time step
    duration = 200.0 # duration of simulation



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
        self.nrn_type = "tonic_spiking" # default
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
        # The input table provides injection current to the neuron.
        # Must be set according to requirements of the test.
        # In particular, xmin, xmax, step_size need to be set.
        self.input = moose.Table("input", self)
        self.input.stepMode = 1 # TAB_LOOP
        self.input.connect("outputSrc", self, "injectDest")
        self.input.connect("output", self.inject_table, "inputRequest")
    
    def schedule(self):
        """Assigns clocks to the model components."""
        self.getContext().setClock(0, SimEnv.dt, 0)
        self.getContext().setClock(1, SimEnv.dt, 1)
        self.inject_table.useClock(0)
        self.vm_table.useClock(1)
        self.inject_table.useClock(1)
        self.useClock(1)

    def set_type(self, name):
        """Parameterizes the model according to its type"""
        global pars
        props = pars[name]
        if props is None:
            print name, ": no such neuron type in dictionary. falling back to tonic spiking."
            return
        self.nrn_type = name
        self.a = props[0]
        self.b = props[1]
        self.c = props[2]
        self.d = props[3]
        self.I = props[4]

    def set_input(self, array):
        """Populate the input table from an array (numpy possibly)"""
        self.input.xmin = 0.0
        self.input.xmax = SimEnv.duration
        self.input.stepSize = SimEnv.dt
        self.input.xdivs = int(SimEnv.duration / SimEnv.dt)
        print len(self.input)
        for i in range(len(array)):
            self.input[i] = array[i]
            
    def init_input(self):
        """Create input according to the type of this neuron."""
        input_array = create_input(self.nrn_type, int(SimEnv.duration / SimEnv.dt))
        self.set_input(input_array)

    def dump_data(self):
        self.vm_table.dumpFile(self.nrn_type + "_vm.plot")
        self.inject_table.dumpFile(self.nrn_type + "_i.plot")
        return self.vm_table

    def fullrun(self):
        self.init_input()
        self.schedule()
        moose.PyMooseBase.getContext().reset()
        moose.PyMooseBase.getContext().step(SimEnv.duration)
        return self.dump_data()

def create_input(nrn_type, input_len):
    if input_len < 200:
        print("Simulate at least for 200 ms.")
        return numpy.zeros(input_len)

    input_array = numpy.zeros(int(input_len))
    if nrn_type == 'tonic_spiking' or  \
            nrn_type == 'phasic_spiking' or \
            nrn_type =="tonic_bursting" or \
            nrn_type =="phasic_bursting" or \
            nrn_type =="mixed_mode" or \
            nrn_type =="spike_freq_adapt":
            input_array[:20] = 0.0
            input_array[20:] = pars[nrn_type][4]
    
    elif nrn_type =="Class_1" or nrn_type =="Class_2":        
        input_array = numpy.linspace(0, 20.0, input_len)
    elif nrn_type =="spike_latency" or nrn_type =="subthresh_osc" or nrn_type =="DAP":
        input_array[20:21] = 20.0
    elif nrn_type =="resonator": 
        isi = 1
        width = 10
        # keep changing the interspike interval to find out the resonance freq
        i = 0
        while i < input_len:
            input_array[i] = 20.0
            index = i + isi + 1
            if index > input_len:
                break
            else:
                input_array[index] = 20.0
            i = i + width
    elif nrn_type =="integrator":     
        input_array[20] = 20.0
        input_array[22] = 20.0
        input_array[100] = 20.0
        input_array[105:110] = 20.0        
    elif nrn_type =="rebound_spike" or  nrn_type =="rebound_burst":
        input_array[20] = -20.0
    elif nrn_type =="thresh_var":     
        input_array[20] = 20.0
        input_array[100] = -20.0
        input_array[102] = 20.0        
    elif nrn_type =="bistable":
        input_array[20] = 20.0
        input_array[100] = 20.0        
    elif nrn_type =="accomodation":   
        input_array[:100] = numpy.linspace(0, 20.0, 100)
        input_array[105] = 5.0
    elif nrn_type =="iispike" or nrn_type =="iiburst":        
        input_array[20:120] = -20.0
    
    return input_array


def run_model(nrn_type):
    nrn = IzhikevichTest(nrn_type)
    nrn.set_type(nrn_type)
    vm_array = numpy.array(nrn.fullrun(SimEnv.duration, SimEnv.dt))
    return (input_array, vm_array)

def run():
    """Runs the simulation."""
    moose.PyMooseBase.getContext().reset()
    moose.PyMooseBase.getContext().step(SimEnv.duration)

def run_all():
    """Run all the models and show the plots.This is for
    matplotlib-only system. No dependency on Qt"""
    global pars
    neurons = []
    data = {}
    nrn = IzhikevichTest("Izhikevich")

    for nrn_type in pars.keys():
        nrn.set_type(nrn_type)
        nrn.schedule(SimEnv.dt)
        neurons.append(nrn)
    run(SimEnv.duration)
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



####################################################
# Qt GUI code starts here
####################################################

from PyQt4 import Qt
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qwt5 as Qwt
import PyQt4.Qwt5.qplt as qplt
import PyQt4.Qwt5.anynumpy as numpy

class IzhikevichGui(QtGui.QMainWindow):
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.setMinimumSize(QtCore.QSize(800, 200))
        self.plots = []
        self.buttons = []
        self.nrn = IzhikevichTest("Izhikevich")
        self.ctrl_frame = QtGui.QFrame(self)
        layout = QtGui.QGridLayout(self.ctrl_frame)
        row = 0
        col = 0
        for key in pars.keys():
            button = QtGui.QPushButton(key, self.ctrl_frame)
            self.connect(button, QtCore.SIGNAL('clicked()'), self.run_slot)
            print row, col
            layout.addWidget(button, row, col)
            print button.text(), row, col
            if col == 0:
                col = 1
            else:
                col = 0
                row = row + 1
            self.buttons.append(button)
        self.ctrl_frame.setLayout(layout)
        self.setCentralWidget(self.ctrl_frame)

    def run_slot(self):
        print "In run slot"
        source = self.sender()
        nrn_type = str(source.text())
        self.nrn.set_type(nrn_type)
        vm_array = numpy.array(self.nrn.fullrun())
        inj_array = numpy.array(self.nrn.inject_table)
        plot = Qwt.QwtPlot()
        plot.setTitle(nrn_type)
        plot.insertLegend(Qwt.QwtLegend(), Qwt.QwtPlot.RightLegend)
        vm_curve = Qwt.QwtPlotCurve(nrn_type + "_Vm")
        vm_curve.setPen(Qt.QPen(Qt.Qt.blue))
        vm_curve.attach(plot)
        inj_curve = Qwt.QwtPlotCurve(plot.tr(nrn_type + "_I"))
        inj_curve.setPen(Qt.QPen(Qt.Qt.red))
        inj_curve.attach(plot)
        vm_curve.setData(numpy.linspace(0, SimEnv.duration, len(vm_array)), vm_array)
        inj_curve.setData(numpy.linspace(0, SimEnv.duration, len(inj_array)), inj_array - 100)
        plot.replot()
        plot.show()
        self.plots.append(plot)
    
if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)
    main_w = IzhikevichGui()
    main_w.show()
    sys.exit(qApp.exec_())
    

# 
# Izhikevich.py ends here
