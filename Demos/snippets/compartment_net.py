# compartment_net.py --- 
# 
# Filename: compartment_net.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Aug 11 14:30:21 2012 (+0530)
# Version: 
# Last-Updated: Sat Aug 11 17:15:51 2012 (+0530)
#           By: subha
#     Update #: 111
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

import moose

EREST_ACT = -65e-3

# Gate equations have the form:
#
# y(x) = (A + B * x) / (C + exp((x + D) / F))
# 
# where x is membrane voltage and y is the rate constant for gate
# closing or opening

Na_m_params = {'A_A':0.1 * (25e-3 + EREST_ACT),
               'A_B': -0.1,
               'A_C': -1.0,
               'A_D': -25e-3 - EREST_ACT,
               'A_F':-10e-3,
               'B_A': 4.0e-3,
               'B_B': 0.0,
               'B_C': 0.0,
               'B_D': 0.0 - EREST_ACT,
               'B_F': 18e-3}
Na_h_params = {'A_A': 0.07e-3,
               'A_B': 0.0,
               'A_C': 0.0,
               'A_D': 0.0 - EREST_ACT,
               'A_F': 20e-3,
               'B_A': 1.0,
               'B_B': 0.0,
               'B_C': 1.0,
               'B_D': -30e-3 - EREST_ACT,
               'B_F': -10e-3}
K_n_params = {'A_A': 0.01*(10e-3 + EREST_ACT),
              'A_B': -0.01,
              'A_C': -1.0,
              'A_D': -10e-3 - EREST_ACT,
              'A_F': -10e-3,
              'B_A': 0.125e-3,
              'B_B': 0.0,
              'B_C': 0.0,
              'B_D': 0.0 - EREST_ACT,
              'B_F': 80e-3}
VMIN = -30e-3 + EREST_ACT
VMAX = 120e-3 + EREST_ACT
VDIVS = 3000

def create_na_proto():
    lib = moose.Neutral('/library')
    na = moose.HHChannel('/library/na')
    na.Xpower = 3
    xGate = moose.element(na.path + '/gateX')    
    xGate.setupAlpha([Na_m_params['A_A'],
                   Na_m_params['A_B'],
                   Na_m_params['A_C'],
                   Na_m_params['A_D'],
                   Na_m_params['A_F'],
                   Na_m_params['B_A'],
                   Na_m_params['B_B'],
                   Na_m_params['B_C'],
                   Na_m_params['B_D'],
                   Na_m_params['B_F'],
                   VDIVS, VMIN, VMAX])
    na.Ypower = 1
    yGate = moose.element(na.path + '/gateY')
    yGate.setupAlpha([Na_m_params['A_A'],
                   Na_h_params['A_B'],
                   Na_h_params['A_C'],
                   Na_h_params['A_D'],
                   Na_h_params['A_F'],
                   Na_h_params['B_A'],
                   Na_h_params['B_B'],
                   Na_h_params['B_C'],
                   Na_h_params['B_D'],
                   Na_h_params['B_F'],
                   VDIVS, VMIN, VMAX])
    return na

def create_k_proto():
    lib = moose.Neutral('/library')
    k = moose.HHChannel('/library/k')
    k.Xpower = 4.0
    xGate = moose.element(k.path + '/gateX')    
    xGate.setupAlpha([K_n_params['A_A'],
                   K_n_params['A_B'],
                   K_n_params['A_C'],
                   K_n_params['A_D'],
                   K_n_params['A_F'],
                   K_n_params['B_A'],
                   K_n_params['B_B'],
                   K_n_params['B_C'],
                   K_n_params['B_D'],
                   K_n_params['B_F'],
                   VDIVS, VMIN, VMAX])
    return k


def create_compartments(path, size):
    comps = moose.Id(path, size, 'Compartment')    
    comps.Em = [-65e-3] * size
    comps.initVm = [-65e-3] * size
    comps.Cm = [1e-12] * size
    comps.Rm = [1e9] * size
    comps.Ra = [1e5] * size
    nachan = moose.copy(create_na_proto(), comps)
    kchan = moose.copy(create_k_proto(), comps)
    synchan = moose.SynChan(path + '/synchan')
    # Question: What is this going to do? Connect comps[ii] to comps[ii]/synapse? 
    m = moose.connect(comps, 'channel', synchan, 'channel', 'OneToOne')
    ## Or is this the correct approach?
    # for c in comps: moose.connect(c, 'channel', moose.SynChan(c.path+'/synchan'), 'channel', 'Single')
    spikegen = moose.SpikeGen(path + '/spikegen')
    m = moose.connect(comps, 'VmOut', spikegen, 'Vm', 'OneToOne')
    ## This only works for synchan[0].
    # synchan.synapse.num = size
    
    ## Actually we have to go through each compartment and update the
    ## synchan under it.
    for c in comps:
        s = moose.SynChan(c.path+'/synchan')
        s.Gbar = 1e-9
        print s.synapse.num
        s.synapse.num = size
    # Another approach could be to make spikegen and synchan outside the compartments and use OneToOne message.
    return comps
    
if __name__ == '__main__':
    comps = create_compartments('/comp', 10)


# 
# compartment_net.py ends here
