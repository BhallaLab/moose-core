# compartment_net.py --- 
# 
# Filename: compartment_net.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Aug 11 14:30:21 2012 (+0530)
# Version: 
# Last-Updated: Sun Aug 12 15:45:38 2012 (+0530)
#           By: subha
#     Update #: 521
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
from pylab import *
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
    xGate = moose.HHGate(na.path + '/gateX')    
    xGate.setupAlpha(Na_m_params +
                      [VDIVS, VMIN, VMAX])
    na.Ypower = 1
    yGate = moose.HHGate(na.path + '/gateY')
    yGate.setupAlpha(Na_h_params + 
                      [VDIVS, VMIN, VMAX])
    return na

def create_k_proto():
    lib = moose.Neutral('/library')
    k = moose.HHChannel('/library/k')
    k.Xpower = 4.0
    xGate = moose.HHGate(k.path + '/gateX')    
    xGate.setupAlpha(K_n_params +
                      [VDIVS, VMIN, VMAX])
    return k

def gate_params(channel):
    """Return a dictionary containing x_inf, y_inf, tau_x, tau_y of
    the specified `channel`.

    If either gate is absent, the corresponding entries in the
    dictionary are empty lists.
    
    """
    xGate = None
    x_inf = []
    tau_x = []
    yGate = None
    y_inf = []
    tau_y = []
    varray = []
    print '>>', channel.path, channel.Xpower, channel.Ypower
    if channel.Xpower > 0:
        xGate = moose.element(channel.path + '/gateX')
        vmin = xGate.min
        vmax = xGate.max
        vdivs = xGate.divs
        varray = linspace(vmin, vmax, vdivs+1)
        print channel.path, vmin, vmax, vdivs
        ax = array([xGate.A[v] for v in varray])
        bx = array([xGate.B[v] for v in varray])
        x_inf = ax/bx
        tau_x = 1/bx
    if channel.Ypower > 0:
        yGate = moose.element(channel.path + '/gateY')
        vmin = yGate.min
        vmay = yGate.max
        vdivs = yGate.divs
        varray = linspace(vmin, vmax, vdivs+1)
        ay = array([yGate.A[v] for v in varray])
        by = array([yGate.B[v] for v in varray])
        y_inf = ay/by
        tau_y = 1/by
    return {'x_inf': x_inf,
            'tau_x': tau_x,
            'y_inf': y_inf,
            'tau_y': tau_y,
            'v_array': varray}

def plot_gate_params(chan):
    """Plot the gate parameters like m and h of the channel."""
    params = gate_params(moose.HHChannel(chan))    
    subplot(2,1,1)    
    plot(params['v_array'], na_params['x_inf'], label='m_inf')
    if len(params['y_inf']) == len(params['v_array']):
        plot(params['v_array'], na_params['y_inf'], label='h_inf')
    legend()
    subplot(212)
    plot(params['v_array'], params['tau_x'], label='tau_m')
    if len(params['y_inf']) == len(params['v_array']):
        plot(params['v_array'], params['tau_y'], label='tau_h')
    legend()
    show()
    
def create_population(container, size):
    """Create a population of `size` single compartmental neurons with
    Na and K channels. Also create SpikeGen objects and SynChan
    objects connected to these which can act as plug points for
    setting up synapses later."""
    path = container.path
    comps = moose.ematrix(path+'/soma', size, 'Compartment')    
    Em = EREST_ACT+10.613e-3
    comps.Em = np.random.normal(Em, np.abs(Em) * 0.1, size)
    comps.initVm = np.random.normal(EREST_ACT, np.abs(EREST_ACT) * 0.1, size)
    comps.Cm = [7.85e-9] * size
    comps.Rm = [4.2e5] * size
    comps.Ra = [190.98] * size
    nachan = moose.copy(create_na_proto(), container, 'na', size)
    nachan.Gbar = [0.942e-3] * size
    nachan.Ek = [115e-3+EREST_ACT] * size
    moose.connect(nachan, 'channel', comps, 'channel', 'OneToOne')
    kchan = moose.copy(create_k_proto(), container, 'k', size)
    kchan.Gbar = [0.2836e-4] * size
    kchan.Ek = [-12e-3+EREST_ACT] * size
    moose.connect(kchan, 'channel', comps, 'channel', 'OneToOne')
    synchan = moose.ematrix(path + '/synchan', size, 'SynChan')
    synchan.Gbar = [1e-8] * size
    synchan.tau1 = [2e-3] * size
    synchan.tau2 = [2e-3] * size
    # Question: What is this going to do? Connect comps[ii] to comps[ii]/synchan? 
    # if we had synchan under each  compartment created like: synchan = moose.SynChan(comps.path + '/synchan')

    m = moose.connect(comps, 'channel', synchan, 'channel', 'OneToOne')

    ## Or would this have been the correct approach?
    # for c in comps: moose.connect(c, 'channel', moose.SynChan(c.path+'/synchan'), 'channel', 'Single')
    spikegen = moose.ematrix(path + '/spikegen', size, 'SpikeGen')
    spikegen.threshold = [0.0] * size
    m = moose.connect(comps, 'VmOut', spikegen, 'Vm', 'OneToOne')
    return {'compartment': comps,
            'spikegen': spikegen,
            'synchan': synchan}

def make_synapses(spikegen, synchan, connprob=1.0, delay=5e-3):
    """Create synapses from spikegen array to synchan array.

    connprob: connection probability.

    delay: mean delay of synaptic transmission. Individual delays are
    normally distributed with sd=0.1*mean.  
    """
    for ii in synchan: 
        s = moose.SynChan(ii)
        scount = len(spikegen)
        s.synapse.num = scount
        delay_list = np.random.normal(delay, delay*0.1, scount)
        for jj in range(scount): s.synapse[jj].delay = delay_list[jj]
    m = moose.connect(spikegen, 'event', moose.element(synchan.path + '/synapse'),  'addSpike', 'Sparse')
    # The sparse message maintains an adjacency matrix. In the special
    # case of synapses on synchan objects, entry a[i][j] = k means
    # that source object no i (say spikegen[i] connects to synapse
    # no. k on the j-th synchan object.
    moose.SparseMsg(m).setRandomConnectivity(connprob, 1)
            
            

if __name__ == '__main__':
    simtime = 0.1
    simdt = 0.25e-5
    plotdt = 0.25e-3
    size = 2
    net = moose.Neutral('network')
    pop_a = create_population(moose.Neutral('/network/pop_A'), size)
    pop_b = create_population(moose.Neutral('/network/pop_B'), size)
    make_synapses(pop_a['spikegen'], pop_b['synchan'])
    pulse = moose.PulseGen('pulse')
    pulse.firstLevel = 1e-9
    pulse.firstDelay = 0.05e10 # disable the pulsegen
    pulse.firstWidth = 1e9
    moose.connect(pulse, 'outputOut', pop_a['compartment'][0], 'injectMsg')
    data = moose.Neutral('/data')
    vm_a = moose.Table('/data/Vm_A', size)
    moose.connect(vm_a, 'requestData', pop_a['compartment'], 'get_Vm', 'OneToOne')
    vm_b = moose.Table('/data/Vm_B', size)
    moose.connect(vm_b, 'requestData', pop_b['compartment'], 'get_Vm', 'OneToOne')
    gksyn_b = moose.Table('/data/Gk_syn_b', size)
    moose.connect(gksyn_b, 'requestData', pop_b['synchan'], 'get_Gk', 'OneToOne')
    pulsetable = moose.Table('/data/pulse')
    pulsetable.connect('requestData', pulse, 'get_output')
    moose.setClock(0, simdt)
    moose.setClock(1, simdt)
    moose.setClock(2, simdt)
    moose.setClock(3, simdt)
    moose.setClock(4, plotdt)
    moose.useClock(0, '/network/##[TYPE=Compartment]', 'init')
    moose.useClock(1, '/network/##[TYPE=Compartment]', 'process')
    moose.useClock(2, '/network/##[TYPE=SynChan]', 'process')
    moose.useClock(2, '/network/##[TYPE=HHChannel]', 'process')
    moose.useClock(3, '/network/##[TYPE=SpikeGen],/#[TYPE=PulseGen]', 'process')    
    moose.useClock(4, '/data/#[TYPE=Table]', 'process')
    moose.reinit()
    moose.start(simtime)
    plt.subplot(221)
    for oid in vm_a.vec:
        plt.plot(oid.vec, label=oid.path)
    plt.legend()
    plt.subplot(223)
    for oid in vm_b.vec:
        plt.plot(oid.vec, label=oid.path)
    plt.legend()
    plt.subplot(224)
    for ii in gksyn_b.vec:
        plt.plot(ii.vec, label=ii.path)
    plt.legend()
    plt.show()
    

# 
# compartment_net.py ends here
