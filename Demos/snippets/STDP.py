#!/usr/bin/env python
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

'''
Simulate a pseudo-STDP protocol and plot the STDP rule.
Author: Aditya Gilra, NCBS, Bangalore, October, 2014.
'''

import moose
from pylab import *

# ###########################################
# Neuron models
# ###########################################

## Leaky integrate and fire neuron
Vrest = -65e-3 # V      # resting potential
Vt_base = -45e-3 # V    # threshold
Vreset = -55e-3 # V     # in current steps, Vreset is same as pedestal
R = 1e8 # Ohm
tau = 10e-3 # s
refrT = 2e-3 # s

# ###########################################
# Initialize neuron group
# ###########################################

## two neurons: index 0 will be presynaptic, 1 will be postsynaptic
network = moose.LIF( 'network', 2 );
moose.le( '/network' )
network.vec.Em = Vrest
network.vec.thresh = Vt_base
network.vec.refractoryPeriod = refrT
network.vec.Rm = R
network.vec.vReset = Vreset
network.vec.Cm = tau/R
network.vec.inject = 0.

# ###########################################
# Synaptic model: STDP at each pre and post spike
# ###########################################

# Values approx from figure in Scholarpedia article (following Bi and Poo 1998):
# Jesper Sjoestroem and Wulfram Gerstner (2010) Spike-timing dependent plasticity.
# Scholarpedia, 5(2):1362., revision #137369
tauPlus = 10e-3 # s         # Apre time constant
tauMinus = 10e-3 # s        # Apost time constant
aPlus0 = 1.0                # at pre, Apre += Apre0
aMinus0 = 0.25              # at post, Apost += Apost0
weight = 5e-3 # V           # delta function synapse, adds to Vm

syn = moose.STDPSynHandler( '/network/syn' )
syn.numSynapses = 1                         # 1 synapse
                                            # many pre-synaptic inputs can connect to a synapse
# synapse onto postsynaptic neuron
moose.connect( syn, 'activationOut', network.vec[1], 'activation' )

# synapse from presynaptic neuron
moose.connect( network.vec[0],'spikeOut', syn.synapse[0], 'addSpike')

# post-synaptic spikes also needed for STDP
moose.connect( network.vec[1], 'spikeOut', syn, 'addPostSpike')

syn.synapse[0].delay = 0.0
syn.synapse[0].weight = weight # V
syn.aPlus0 = aPlus0*weight      # on every pre-spike, aPlus gets this jump
                                # aMinus0 includes learning rate
                                # on every pre-spike, aMinus is added to weight
syn.tauPlus = tauPlus
syn.aMinus0 = -aMinus0*weight   # on every post-spike, aMinus gets this jump
                                # aMinus0 includes learning rate
                                # on every post-spike, aPlus is added to weight
syn.tauMinus = tauMinus
syn.weightMax = 2*weight        # bounds on the weight
syn.weightMin = 0.

# ###########################################
# Setting up tables
# ###########################################

Vms = moose.Table( '/plotVms', 2 )
moose.connect( network, 'VmOut', Vms, 'input', 'OneToOne')
spikes = moose.Table( '/plotSpikes', 2 )
moose.connect( network, 'spikeOut', spikes, 'input', 'OneToOne')

# ###########################################
# Simulate the STDP curve with spaced pre-post spike pairs
# ###########################################

dt = 0.5e-3 # s
# moose simulation
moose.useClock( 0, '/network/syn', 'process' )
moose.useClock( 1, '/network', 'process' )
moose.useClock( 2, '/plotSpikes', 'process' )
moose.useClock( 3, '/plotVms', 'process' )
moose.setClock( 0, dt )
moose.setClock( 1, dt )
moose.setClock( 2, dt )
moose.setClock( 3, dt )
moose.setClock( 9, dt )


##  function to make the aPlus and aMinus settle to equilibrium values
settletime = 100e-3 # s
def reset_settle():
    """ Call this between every pre-post pair
    to reset the neurons and make them settle to rest.
    """
    syn.synapse[0].weight = weight # V
    moose.start(settletime)

def make_neuron_spike(nrnidx,I=1e-7,duration=1e-3):
    """ Inject a brief current pulse to 
    make a neuron spike
    """
    network.vec[nrnidx].inject = I
    moose.start(duration)
    network.vec[nrnidx].inject = 0.

dwlist_neg = []
ddt = 2e-3 # s
t_extent = 20e-3 # s
# dt = tpost - tpre
# negative dt corresponds to post before pre
print '-----------------------------------------------'
for deltat in arange(t_extent,0.0,-ddt):
    reset_settle()
    # post neuron spike
    make_neuron_spike(1)
    moose.start(deltat)
    # pre neuron spike after deltat
    make_neuron_spike(0)
    moose.start(1e-3)
    dw = ( syn.synapse[0].weight - weight ) / weight
    print 'post before pre, dt = %1.3f s, dw/w = %1.3f'%(-deltat,dw)
    dwlist_neg.append(dw)
print '-----------------------------------------------'
# positive dt corresponds to pre before post
dwlist_pos = []
for deltat in arange(ddt,t_extent+ddt,ddt):
    reset_settle()
    # pre neuron spike
    make_neuron_spike(0)
    moose.start(deltat)
    # post neuron spike after deltat
    make_neuron_spike(1)
    moose.start(1e-3)
    dw = ( syn.synapse[0].weight - weight ) / weight
    print 'pre before post, dt = %1.3f s, dw/w = %1.3f'%(deltat,dw)
    dwlist_pos.append(dw)
print '-----------------------------------------------'

# ###########################################
# Plot the simulated Vm-s and STDP curve
# ###########################################

# Voltage plots
# insert spikes from Spike Monitor so that Vm doesn't look weird
figure(facecolor='w')
plot(Vms.vec[0].vector,color='r') # pre neuron's vm
plot(Vms.vec[1].vector,color='b') # post neuron's vm
xlabel('time (s)')
ylabel('Vm (V)')
title("pre (r) and post (b) neurons' Vm")

# STDP curve
fig = figure(facecolor='w')
ax = fig.add_subplot(111)
ax.plot(arange(-t_extent,0,ddt),array(dwlist_neg),'.-r')
ax.plot(arange(ddt,(t_extent+ddt),ddt),array(dwlist_pos),'.-b')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax.set_xticks([xmin,0,xmax])
ax.set_yticks([ymin,0,ymax])
ax.plot((0,0),(ymin,ymax),linestyle='dashed',color='k')
ax.plot((xmin,xmax),(0,0),linestyle='dashed',color='k')
ax.set_xlabel('$t_{post}-t_{pre}$ (s)')
ax.set_ylabel('$\Delta w / w$')
fig.tight_layout()
#fig.subplots_adjust(hspace=0.3,wspace=0.5) # has to be after tight_layout()

show()
