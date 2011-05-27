# test_nmda.py --- 
# 
# Filename: test_nmda.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Mar 22 16:58:57 2010 (+0530)
# Version: 
# Last-Updated: Fri May 27 11:09:34 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 163
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This code is for testing the NMDAChan in MOOSE - the comparison is
# traub_nmda.mod in neuron version of Traub et al 2005 model.
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

import moose

def testNMDAChan(simtime=50e-3, simdt=1e-4):
    context = moose.PyMooseBase.getContext()
    container = moose.Neutral('test_NMDA')
    soma_a = moose.Compartment('A', container)
    soma_a.Rm = 5.3e9 # GM = 2e-5 S/cm^2
    soma_a.Cm = 8.4823001646924426e-12 # CM = 0.9 uF/cm^2
    soma_a.Em = -65e-3
    soma_a.initVm = -65e-3
    soma_a.Ra = 282942.12
    
    soma_b = moose.Compartment('B', container)
    soma_b.Rm = 5.3e9 # GM = 2e-5 S/cm^2
    soma_b.Cm = 8.4823001646924426e-12 # CM = 0.9 uF/cm^2
    soma_b.Em = -65e-3 
    soma_b.initVm = -65e-3
    soma_b.Ra = 282942.12 # RA = 250 Ohm-cm
    
    nmda = moose.NMDAChan('nmda', container)
    nmda.Gbar = 1.0 # Default
    nmda.Mg = 1.5 # mM = Mol/m^3  - SI :) 
    nmda.tau1 = 130.5e-3 
    nmda.tau2 = 5e-3 
    nmda.connect('channel', soma_b, 'channel')
    
    spikegen = moose.SpikeGen('spike', container)
    spikegen.threshold = 0.0
    soma_a.connect('VmSrc', spikegen, 'Vm')
    spikegen.connect('event', nmda, 'synapse')
    # Only after setting up connections we have access to individual
    # delay and weight for each connection

    # The weight is set to 0.25e-3 uS in test_nmda.hoc
    nmda.weight[0] = 2.5e-10
    nmda.delay[0] = 0.05e-3

    pulsegen = moose.PulseGen('pulse', container)
    pulsegen.firstLevel = 0.1e-9
    pulsegen.firstDelay = 20e-3
    pulsegen.firstWidth = 20e-3
    pulsegen.connect('outputSrc', soma_a, 'injectMsg')

    data = moose.Neutral('data')
    vmB = moose.Table('Vm_B', data)
    vmB.stepMode = 3
    vmB.connect('inputRequest', soma_b, 'Vm')

    vmA = moose.Table('Vm_A', data)
    vmA.stepMode = 3
    vmA.connect('inputRequest', soma_a, 'Vm')

    gNMDA = moose.Table('G', data)
    gNMDA.stepMode = 3
    gNMDA.connect('inputRequest', nmda, 'Gk')

    context.setClock(0, simdt)
    context.setClock(1, simdt)
    context.setClock(2, simdt)
    context.setClock(3, simdt)
    context.reset()

    context.step(simtime)
    outfile = open('nmda_moose.plot', 'w')
    ms = 1e-3
    mV = 1e-3
    uS = 1e-6
    for ii in range(len(gNMDA)):
        outfile.write('%g\t%g\t%g\t%g\n' % (ii*simdt/ms, vmA[ii]/mV, vmB[ii]/mV, gNMDA[ii]/uS))
    outfile.close()
if __name__ == '__main__':
    testNMDAChan()

    
    

# 
# test_nmda.py ends here
