# compartment_net.py --- 
# 
# Filename: compartment_net.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Aug 11 14:30:21 2012 (+0530)
# Version: 
# Last-Updated: Sun Aug 12 01:06:14 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 294
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# A demo to create a network of single compartmental neurons connected
# via alpha synapses.
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
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt

import moose

EREST_ACT = -70e-3

# Gate equations have the form:
#
# y(x) = (A + B * x) / (C + exp((x + D) / F))
# 
# where x is membrane voltage and y is the rate constant for gate
# closing or opening

Na_m_params = [1e5 * (25e-3 + EREST_ACT),   # 'A_A':
                -1e5,                       # 'A_B':
                -1.0,                       # 'A_C':
                -25e-3 - EREST_ACT,         # 'A_D':
               -10e-3,                      # 'A_F':
                4e3,                     # 'B_A':
                0.0,                        # 'B_B':
                0.0,                        # 'B_C':
                0.0 - EREST_ACT,            # 'B_D':
                18e-3                       # 'B_F':    
               ]
Na_h_params = [ 70.0,                        # 'A_A':
                0.0,                       # 'A_B':
                0.0,                       # 'A_C':
                0.0 - EREST_ACT,           # 'A_D':
                0.02,                     # 'A_F':
                1000.0,                       # 'B_A':
                0.0,                       # 'B_B':
                1.0,                       # 'B_C':
                -30e-3 - EREST_ACT,        # 'B_D':
                -0.01                    # 'B_F':       
                ]        
K_n_params = [ 1e4 * (10e-3 + EREST_ACT),   #  'A_A':
               -1e4,                      #  'A_B':
               -1.0,                       #  'A_C':
               -10e-3 - EREST_ACT,         #  'A_D':
               -10e-3,                     #  'A_F':
               0.125e3,                   #  'B_A':
               0.0,                        #  'B_B':
               0.0,                        #  'B_C':
               0.0 - EREST_ACT,            #  'B_D':
               80e-3                       #  'B_F':  
               ]
VMIN = -30e-3 + EREST_ACT
VMAX = 120e-3 + EREST_ACT
VDIVS = 3000

def create_na_proto():
    lib = moose.Neutral('/library')
    na = moose.HHChannel('/library/na')
    na.Xpower = 3
    xGate = moose.element(na.path + '/gateX')    
    xGate.setupAlpha(Na_m_params +
                      [VDIVS, VMIN, VMAX])
    na.Ypower = 1
    yGate = moose.element(na.path + '/gateY')
    yGate.setupAlpha(Na_h_params + 
                      [VDIVS, VMIN, VMAX])
    varray = np.linspace(VMIN, VMAX, VDIVS+1)
    ax = np.array([xGate.A[v] for v in varray])
    bx = np.array([xGate.B[v] for v in varray])
    plt.subplot(2,2,1)
    plt.plot(ax/bx, label='m_inf')
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(1/bx, label='tau_m')
    plt.legend()
    ax = np.array([yGate.A[v] for v in varray])
    bx = np.array([yGate.B[v] for v in varray])
    plt.subplot(2,2,3)
    plt.plot(ax/bx, label='h_inf')
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(1/bx, label='tau_h')
    plt.legend()
    plt.show()
    return na

def create_k_proto():
    lib = moose.Neutral('/library')
    k = moose.HHChannel('/library/k')
    k.Xpower = 4.0
    xGate = moose.element(k.path + '/gateX')    
    xGate.setupAlpha(K_n_params +
                      [VDIVS, VMIN, VMAX])
    varray = np.linspace(VMIN, VMAX, VDIVS+1)
    ax = np.array([xGate.A[v] for v in varray])
    bx = np.array([xGate.B[v] for v in varray])
    plt.subplot(2,1,1)
    plt.plot(ax/bx, label='n_inf')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(1/bx, label='tau_n')
    plt.legend()
    plt.show()
    return k

def create_network(container, size):
    path = container.path
    comps = moose.Id(path+'/soma', size, 'Compartment')    
    comps.Em = [EREST_ACT+10.613e-3] * size
    comps.initVm = [-65e-3] * size
    comps.Cm = [1e-12] * size
    comps.Rm = [1e9] * size
    comps.Ra = [1e5] * size
    nachan = moose.copy(create_na_proto(), comps[0])
    moose.connect(nachan, 'channel', comps, 'channel')
    # moose.connect(nachan, 'channel', comps, 'channel', 'OneToOne')
    kchan = moose.copy(create_k_proto(), comps[0])
    moose.connect(kchan, 'channel', comps, 'channel')
    # moose.connect(kchan, 'channel', comps, 'channel', 'OneToOne')
    synchan = moose.Id(path + '/synchan', size, 'SynChan')
    synchan.Gbar = [1e-9] * size
    for s in synchan: moose.SynChan(s).synapse.num = size
    # Question: What is this going to do? Connect comps[ii] to comps[ii]/synchan? 
    # if we had synchan under each  compartment created like: synchan = moose.SynChan(comps.path + '/synchan')
    m = moose.connect(comps, 'channel', synchan, 'channel', 'OneToOne')
    ## Or would this have been the correct approach?
    # for c in comps: moose.connect(c, 'channel', moose.SynChan(c.path+'/synchan'), 'channel', 'Single')
    spikegen = moose.Id(path + '/spikegen', size, 'SpikeGen')
    spikegen.threshold = [0.0] * size
    m = moose.connect(comps, 'VmOut', spikegen, 'Vm', 'OneToOne')
    # m = moose.connect(spikegen, 'event', moose.ObjId(synchan.path + '/synapse'),  'addSpike', 'Sparse')
    # m.setRandomConnectivity(0.2, 0)
    return {'compartment': comps,
            'spikegen': spikegen,
            'synchan': synchan}
            
            

if __name__ == '__main__':
    simtime = 0.1
    simdt = 0.25e-5
    plotdt = 0.25e-3
    size = 1
    net = create_network(moose.Neutral('network'), size)
    pulse = moose.PulseGen('pulse')
    pulse.firstLevel = 1e-12
    pulse.firstDelay = 0.05
    pulse.firstWidth = 1e9
    moose.connect(pulse, 'outputOut', net['compartment'][0], 'injectMsg')
    data = moose.Neutral('/data')
    vmtables = moose.Table('/data/Vm', size)
    # moose.connect(tables, 'requestData', net['compartment'], 'get_Vm', 'OneToOne')    
    moose.connect(vmtables, 'requestData', net['compartment'], 'get_Vm')
    pulsetable = moose.Table('/data/pulse')
    pulsetable.connect('requestData', pulse, 'get_output')
    moose.setClock(0, simdt)
    moose.setClock(1, simdt)
    moose.setClock(2, simdt)
    moose.setClock(3, simdt)
    moose.setClock(4, plotdt)
    moose.useClock(0, '/network/#[TYPE=Compartment]', 'init')
    moose.useClock(1, '/network/#[TYPE=Compartment]', 'process')
    moose.useClock(2, '/network/##[TYPE=SynChan],/network/##[TYPE=HHChannel]', 'process')
    moose.useClock(3, '/network/#[TYPE=SpikeGen],/#[TYPE=PulseGen]', 'process')    
    moose.useClock(4, '/data/#[TYPE=Table]', 'process')
    moose.reinit()
    moose.start(simtime)
    for oid in vmtables.id_:
        print oid.path, len(oid.vec)
        plt.plot(oid.vec, label=oid.path)
    plt.plot(pulsetable.vec, label='Pulse')
    plt.legend()
    plt.show()
    

# 
# compartment_net.py ends here
