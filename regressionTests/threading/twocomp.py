# twocomp.py --- 
# 
# Filename: twocomp.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Oct  1 17:34:37 2012 (+0530)
# Version: 
# Last-Updated: Tue Oct  2 17:11:00 2012 (+0530)
#           By: subha
#     Update #: 37
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

import pylab
import moose
from datetime import datetime

if __name__ == '__main__':
    # Create the somatic compartment
    model = moose.Neutral('/model') # This is a container for the model
    soma = moose.Compartment('/model/soma')
    soma.Em = -65e-3 # Leak potential
    soma.initVm = -65e-3 # Initial membrane potential
    soma.Rm = 5e9 # Total membrane resistance of the compartment
    soma.Cm = 1e-12 # Total membrane capacitance of the compartment
    soma.Ra = 1e6 # Total axial resistance of the compartment
    # Create the axon
    axon = moose.Compartment('/model/axon')
    axon.Em = -65e-3
    axon.initVm = -65e-3
    axon.Rm = 2.5e9
    axon.Cm = 2e-12
    axon.Ra = 2e5
    # Connect the soma to the axon. Note the order of raxial-axial
    # connection decides which Ra is going to be used in the computation.
    # `raxial` message sends the Ra and Vm of the source to destionation,
    # `axial` gets back the Vm of the destination. Try:
    # moose.doc('Compartment.axial') in python interpreter for details.
    moose.connect(soma, 'raxial', axon, 'axial')
    pulsegen = moose.PulseGen('/model/pulse')
    pulsegen.delay[0] = 50e-3
    pulsegen.level[0] = 1e-12
    pulsegen.width[0] = 100e-3
    pulsegen.delay[1] = 1e9
    moose.connect(pulsegen, 'outputOut', soma, 'injectMsg')
    # Setup data recording
    data = moose.Neutral('/data')
    axon_Vm = moose.Table('/data/axon_Vm')
    moose.connect(axon_Vm, 'requestData', axon, 'get_Vm')
    # Now schedule the sequence of operations and time resolutions
    moose.setClock(0, 0.025e-3)
    moose.setClock(1, 0.025e-3)
    moose.setClock(2, 0.25e-3)
    # useClock: First argument is clock no.
    # Second argument is a wildcard path matching all elements of type Compartment
    # Last argument is the processing function to be executed at each tick of clock 0 
    moose.useClock(0, '/model/#[TYPE=Compartment]', 'init') 
    moose.useClock(1, '/model/#', 'process')
    moose.useClock(2, axon_Vm.path, 'process')
    # Now initialize everything and get set
    moose.reinit()
    start = datetime.now()
    moose.start(300e-3)
    end = datetime.now()
    delta = end - start
    print 'Simulation time (excluding reinit): %g s' % (delta.seconds + delta.microseconds * 1e-6)

# 
# twocomp.py ends here
