# spikegen.py --- 
# 
# Filename: spikegen.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Oct 22 10:02:17 2009 (+0530)
# Version: 
# Last-Updated: Thu Oct 22 13:53:09 2009 (+0530)
#           By: subhasis ray
#     Update #: 176
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
#
# This is a demo for spikegen. There are two kinds of behavior you can
# get from a MOOSE spikegen object. 
#
# 1. GENESIS compatible version: When you set refractT >=
# 0.0, the spikegen will generate a spike as soon as Vm crosses
# threshold and wait until refractT time has passed. If the Vm is
# still above threshold, it will generate another spike. Thus, if the
# mebrane potential(Vm) remains high for more than refractT, this will
# send out multiple spikes.
# 
# 2. NEURON compatible version: When you set refractT < 0.0, the
# spikegen will generate a spike when Vm goes above threshold and then
# become inactive until Vm goes below threshold again. Thus, it
# generates a single spike at the rising edge of Vm.
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
sys.path.append('/home/subha/src/moose/pymoose')
# sys.path.append('/home/subha/src/sim/cortical/py')
import moose
from math import *
dt = 1e-5
context = moose.PyMooseBase.getContext()
def setup(container='/', dt=1e-5):
    comp = moose.Compartment('comp')
    print comp.path
    comp.length = 20e-6
    comp.diameter = 2 * 7.5e-6
    comp.xarea = pi * comp.diameter * comp.diameter / 4.0
    comp.sarea = pi * comp.diameter * comp.length
    comp.Rm = 1.0 / comp.sarea			# specific rm = 1.0 Ohm-m^2
    comp.Ra = 2.5 * comp.length / comp.xarea	# specific ra = 2.5 Ohm-m
    comp.Cm = 1e-3 * comp.sarea			# spcific cm = 1e-3 F/m^2
    comp.Em = -70e-3
    comp.initVm = -70e-3
    print comp.Em, comp.initVm, comp.Vm
    pulsegen = moose.PulseGen('pulsegen')
    pulsegen.firstLevel = 100e-12	# 100 pA current
    pulsegen.firstWidth = 20e-3		# each pulse 20 ms wide
    pulsegen.firstDelay = 20e-3		# pulses every 20 ms

    # This is the GENESIS compatible spike:
    # It will keep spiking until Vm goes below threshold = -40 mV
    spikegen_a = moose.SpikeGen('spike_a')
    spikegen_a.refractT = 2e-3	# 2 ms absolute refractory period
    spikegen_a.threshold = -40e-3
    
    # This is NEURON style spikegen: It will send out a spike when Vm goes
    # above threshold and then beome inactive until Vm returns
    # subthreshold value.
    spikegen_b = moose.SpikeGen('spike_b')
    spikegen_b.refractT = -10.0 # This decides the behaviour
    spikegen_b.threshold = -40e-3
    
    # recording tables:
    pulse_table = moose.Table('pulse')
    vm_table = moose.Table('Vm')
    a_table = moose.Table('SpikeA')
    b_table = moose.Table('SpikeB')
    pulse_table.stepMode = 3
    vm_table.stepMode = 3
    a_table.stepMode = 3
    b_table.stepMode = 3
    
    pulsegen.connect('output', pulse_table, 'inputRequest')
    comp.connect('Vm', vm_table, 'inputRequest')
    spikegen_a.connect('state', a_table, 'inputRequest')
    spikegen_b.connect('state', b_table, 'inputRequest')
    comp.connect('VmSrc', spikegen_a, 'Vm')
    comp.connect('VmSrc', spikegen_b, 'Vm')
    pulsegen.connect('outputSrc', comp, 'injectMsg')

    context.setClock(0,dt)
    context.setClock(1,dt)
    context.reset()
    print comp.Em, comp.initVm, comp.Vm
    
    return [pulse_table, vm_table, a_table, b_table]

import pylab
if __name__ == '__main__':
    simtime = 100e-3
    tables = setup('/')
    context.step(simtime)
    t = pylab.linspace(0, simtime, len(tables[0]))
    for table in tables:
	table.dumpFile(table.name)
    pylab.plot(t, pylab.array(tables[0])*1e7, 'k-', label='pulse')
    pylab.plot(t, pylab.array(tables[1])*1e-2, 'g-', label='vm')
    pylab.plot(t, pylab.array(tables[2])*1e-3, 'b-', label='a')
    pylab.plot(t, pylab.array(tables[3])*1e-3, 'r-.', label='b')
    pylab.legend()
    pylab.show()

# 
# spikegen.py ends here
