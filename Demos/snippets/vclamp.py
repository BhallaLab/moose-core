# vclamp.py --- 
# 
# Filename: vclamp.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Feb  2 19:16:54 2013 (+0530)
# Version: 
# Last-Updated: Mon Feb  4 19:14:45 2013 (+0530)
#           By: subha
#     Update #: 172
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
sys.path.append('/home/subha/src/moose/python')
import moose
sys.path.append('../squid')
from squid import SquidAxon
from pylab import *

def vclamp_demo(simtime=1000.0, dt=1e-2):
    # It is good practice to modularize test elements inside a
    # container
    container = moose.Neutral('/vClampDemo')
    clamp = moose.VClamp('/vClampDemo/vclamp')
    # Setup command voltage time course
    command = moose.PulseGen('/vClampDemo/command')
    command.delay[0] = 200.0
    command.width[0] = 100.0
    command.level[0] = -10.0
    command.delay[1] = 500.0
    command.level[1] = 55.0
    command.width[1] = 100.0
    moose.connect(command, 'outputOut', clamp, 'set_holdingPotential')
    # Create a compartment with properties of a squid giant axon
    comp = SquidAxon('/vClampDemo/axon')
    # Connect the Voltage Clamp to the compartemnt
    moose.connect(clamp, 'currentOut', comp, 'injectMsg')
    moose.connect(comp, 'VmOut', clamp, 'voltageIn')
    clamp.gain = comp.Cm/dt # This is a decent gain value
    # # This is for checking a current clamp
    # stim = moose.PulseGen('/vClampDemo/stim')
    # stim.delay[0] = 1e9
    # stim.width[0] = 100.0
    # stim.level[0] = 0.1
    # stim.delay[1] = 500.0
    # stim.level[1] = 0.2
    # stim.width[1] = 1e9
    # moose.connect(stim, 'outputOut', comp, 'injectMsg')
    # setup stimulus recroding
    # stimtab = moose.Table('/vClampDemo/stimtab')
    # moose.connect(stimtab, 'requestData', stim, 'get_output')
    # Set up Vm recording
    vmtab = moose.Table('/vClampDemo/vclamp_Vm')
    moose.connect(vmtab, 'requestData', comp, 'get_Vm')
    # setup command potential recording
    commandtab = moose.Table('/vClampDemo/vclamp_command')
    moose.connect(commandtab, 'requestData', clamp, 'get_holdingPotential')
    # setup current recording
    Imtab = moose.Table('/vClampDemo/vclamp_Im')
    moose.connect(Imtab, 'requestData', clamp, 'get_current')
    # Scheduling
    moose.setClock(0, dt)
    moose.setClock(1, dt)    
    moose.setClock(2, dt)
    moose.setClock(3, dt)
    moose.useClock(0, '%s/##[TYPE=Compartment]' % (container.path), 'init')
    moose.useClock(0, '%s/##[TYPE=PulseGen]' % (container.path), 'process')
    moose.useClock(1, '%s/##[TYPE=Compartment]' % (container.path), 'process')
    moose.useClock(2, '%s/##[TYPE=HHChannel]' % (container.path), 'process')
    moose.useClock(2, '%s/##[TYPE=VClamp]' % (container.path), 'process')
    moose.useClock(3, '%s/##[TYPE=Table]' % (container.path), 'process')
    moose.reinit()
    moose.start(simtime)
    print 'Finished simulation for %g seconds' % (simtime)
    tseries = linspace(0, simtime, len(vmtab.vec))
    subplot(211)
    title('Membrane potential and clamp voltage')
    plot(tseries, vmtab.vec, label='Vm (mV)')
    plot(tseries, commandtab.vec, label='Command (mV)')
    legend()
    # print len(commandtab.vec)
    subplot(212)
    title('Current through clamp circuit')
    # plot(tseries, stimtab.vec, label='stimulus (uA)')
    plot(tseries, Imtab.vec, label='Im (uA)')
    legend()
    show()

if __name__ == '__main__':
    vclamp_demo()
    

# 
# vclamp.py ends here
