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
Simulate current injection into various Integrate and Fire neurons.
Author: Aditya Gilra, NCBS, Bangalore, October, 2014.
'''

import moose
from pylab import *

neuronChoices = {'LIF':moose.LIF, 'QIF':moose.QIF, 'ExIF':moose.ExIF, 'AdExIF':moose.AdExIF}
choiceIndex = 'AdExIF'
neuronChoice = neuronChoices[choiceIndex]

# ###########################################
# Neuron model parameters
# ###########################################
#
# LIF:      Rm*Cm dVm/dt = -(Vm-Em) + Rm*I
# QIF:      Rm*Cm dVm/dt = a0*(Vm-Em)*(Vm-vCritical) + Rm*I
# ExIF:    Rm*Cm dVm/dt = -(Vm-Em) + deltaThresh * exp((Vm-thresh)/deltaThresh) + Rm*I
# AdExIF    Rm*Cm dVm/dt = -(Vm-Em) + deltaThresh * exp((Vm-thresh)/deltaThresh) + Rm*I - w
#           tau_w dw/dt = a0*(Vm-Em) - w
#
# ###########################################

# Quadratic leaky Integrate and Fire neuron
Vrest = -65e-3 # V      # resting potential
Vt_base = -45e-3 # V    # threshold
Vreset = -55e-3 # V     # Vreset need not be same as Vrest
R = 1e8 # Ohm
tau = 10e-3 # s
refrT = 2e-3 # s
# for QIF
vCritical = -54e-3 # V  # critical voltage above
                        # which Vm rises fast quadratically
a0 = 1e3 # V^-1         # parameter in equation
# for ExIF
deltaThresh = 5e-3 # V
vPeak = 30e-3 # V       # for ExpIF reset is from vPeak, not thresh
                        # I also use vPeak for adding spikes post-simulation
                        # to LIF, QIF, etc.
# for AdExIF
a0AdEx = 0.0 # unitless # voltage-dependent adaptation factor
b0 = 5e-10 # Amp        # current step added to the adaptation current
tauW = 20e-3 # s        # decay time constant of the adaptation current

# ###########################################
# Initialize neuron group
# ###########################################

# neuron instantiation
network = neuronChoice( 'network' ); # choose neuron type above
moose.le( '/network' )
network.vec.Em = Vrest
network.vec.thresh = Vt_base
network.vec.refractoryPeriod = refrT
network.vec.Rm = R
network.vec.vReset = Vreset
network.vec.Cm = tau/R
network.vec.initVm = Vrest

# neuron specific parameters and current injected I
if choiceIndex == 'LIF':
    network.vec.inject = 5e-10 # Amp    # injected current I
if choiceIndex == 'QIF':
    network.vec.a0 = a0
    network.vec.vCritical = vCritical
    network.vec.inject = 5e-10 # Amp    # injected current I
elif choiceIndex == 'ExIF':
    network.vec.deltaThresh = deltaThresh
    network.vec.inject = 1e-9  # Amp    # injected current I
elif choiceIndex == 'AdExIF':
    network.vec.deltaThresh = deltaThresh
    network.vec.a0 = a0AdEx
    network.vec.b0 = b0
    network.vec.tauW = tauW
    network.vec.inject = 1e-9  # Amp    # injected current I

# ###########################################
# Setting up table
# ###########################################

Vm = moose.Table( '/plotVm' )
moose.connect( network, 'VmOut', Vm, 'input', 'OneToOne')
spikes = moose.Table( '/plotSpikes' )
moose.connect( network, 'spikeOut', spikes, 'input', 'OneToOne')

# ###########################################
# Simulate the current injection
# ###########################################

dt = 5e-6 # s
runtime = 0.02 # s

# moose simulation
moose.useClock( 1, '/network', 'process' )
moose.useClock( 2, '/plotSpikes', 'process' )
moose.useClock( 3, '/plotVm', 'process' )
moose.setClock( 0, dt )
moose.setClock( 1, dt )
moose.setClock( 2, dt )
moose.setClock( 3, dt )
moose.setClock( 9, dt )
moose.reinit()
moose.start(runtime)

# ###########################################
# Plot the simulated Vm-s and STDP curve
# ###########################################

# Voltage plots
# insert spikes so that Vm reset doesn't look weird
Vmseries = Vm.vector
numsteps = len(Vmseries)
for t in spikes.vector:
    Vmseries[int(t/dt)-1] = 30e-3 # V
timeseries = arange(0.,1000*numsteps*dt-1e-10,dt*1000)
figure(facecolor='w')
plot(timeseries,Vmseries,color='r') # neuron's Vm
xlabel('time (ms)')
ylabel('Vm (V)')
title(choiceIndex)

show()
