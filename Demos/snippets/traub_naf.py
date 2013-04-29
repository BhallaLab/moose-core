# traub_naf.py --- 
# 
# Filename: traub_naf.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Apr 29 21:07:30 2013 (+0530)
# Version: 
# Last-Updated: Mon Apr 29 21:55:30 2013 (+0530)
#           By: subha
#     Update #: 127
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

"""This is an example showing pymoose implementation of the NaF
channel in Traub et al 2005

Author: Subhasis Ray

"""

import numpy as np
import pylab
import moose
from moose import utils

vmin = -120e-3
vmax = 40e-3
vdivs = 640
v_array = np.linspace(vmin, vmax, vdivs+1)

def create_naf_proto():
    """Create an NaF channel prototype in /library. You can copy it later
    into any compartment or load a .p file with this channel using
    loadModel.
    
    This channel has the conductance form:

    Gk(v) = Gbar * m^3 * h (V - Ek)

    We are using all SI units

    """
    if moose.exists('/library/NaF'):
        return moose.element('/library/NaF')
    if not moose.exists('/library'):
        lib = moose.Neutral('/library')
    channel = moose.HHChannel('/library/NaF')
    shift = -3.5e-3
    # tau_m is defined piecewise:
    # tau_m = 1.0e-3 * (0.025 + 0.14 * exp(( v + shift + 30e-3) / 10)) if v + shift < -30e-3
    #       = 1.0e-3 * (0.02 + 0.145 * np.exp(( - v_array - shift - 30.0e-3) / 10.0e-3)) otherwise
    tau_m = np.where((v_array + shift) < -30e-3,
                     1.0e-3 * (0.025 + 0.14 * np.exp((v_array + shift + 30.0e-3) / 10.0e-3)), \
                     1.0e-3 * (0.02 + 0.145 * np.exp(( - v_array - shift - 30.0e-3) / 10.0e-3)))
    inf_m = 1.0 / (1.0 + np.exp(( - v_array - shift - 38e-3) / 10e-3))
    tau_h = 1.0e-3 * (0.15 + 1.15 / ( 1.0 + np.exp(( v_array + 37.0e-3) / 15.0e-3)))
    inf_h = 1.0 / (1.0 + np.exp((v_array + 62.9e-3) / 10.7e-3))
    channel.Xpower = 3 # Creates m-gate
    # In svn version of moose you can even do:
    # mgate = channel.gateX[0]
    mgate = moose.element('%s/gateX' % (channel.path))
    mgate.tableA = inf_m / tau_m
    mgate.tableB = 1 / tau_m
    channel.Ypower = 1 # Creates h-gate
    hgate = moose.element('%s/gateY' % (channel.path))
    hgate.tableA = inf_h / tau_h
    hgate.tableB = 1 / tau_h
    return channel
    
def create_compartment(parent_path, name):    
    """This shows how to use the prototype channel on a compartment."""
    comp = moose.Compartment('%s/%s' % (parent_path, name))
    comp.Rm = 1e6
    comp.Ra = 1e9
    comp.Cm = 1e-8
    comp.initVm = -0.06
    comp.Em = -0.06
    protochan = create_naf_proto()
    chan = moose.copy(protochan, comp, 'NaF')
    chan.Gbar = 1e-9
    moose.connect(comp, 'channel', chan, 'channel')
    return comp

if __name__ == '__main__':
    model = moose.Neutral('/model')
    comp = create_compartment(model.path, 'soma')    
    vclamp = moose.VClamp('%s/vclamp' % (model.path))
    command = moose.PulseGen('%s/command' % (model.path))
    moose.connect(command, 'outputOut', vclamp, 'set_command')
    moose.connect(comp, 'VmOut', vclamp, 'set_sensed')
    moose.connect(vclamp, 'currentOut', comp, 'injectMsg')
    command.delay[0] = 20e-3
    command.width[0] = 50e-3
    command.level[0] = 0.010
    
    data =moose.Neutral('/data')
    Iinj = moose.Table('%s/Iinj' % (data.path))
    moose.connect(Iinj, 'requestData', vclamp, 'get_current')
    Vm = moose.Table('%s/Vm' % (data.path))
    moose.connect(Vm, 'requestData', comp, 'get_Vm')

    simtime = 100e-3
    simdt = 1e-6
    plotdt = 1e-4
    solver = 'ee'
    utils.resetSim([model.path, data.path], simdt, plotdt, simmethod=solver)
    moose.start(simtime)

    ivec = np.asarray(Iinj.vec)
    vvec = np.asarray(Vm.vec)
    ts = np.linspace(0, simtime, len(vvec))
    
    pylab.subplot(2,1,1)
    pylab.plot(ts, ivec)
    pylab.ylabel('Injected current (A)')
    pylab.subplot(2, 1, 2)
    pylab.ylabel('Membrane voltage (V)')
    pylab.plot(ts, vvec)
    pylab.show()

# 
# traub_naf.py ends here
