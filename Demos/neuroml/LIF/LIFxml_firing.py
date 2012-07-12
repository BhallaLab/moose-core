# -*- coding: utf-8 -*-
## all SI units
########################################################################################
## Plot the membrane potential for a leaky integrate and fire neuron with current injection
## Author: Aditya Gilra
## Creation Date: 2012-06-08
## Modification Date: 2012-06-08
########################################################################################

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')

## simulation parameters
SIMDT = 5e-5 # seconds
PLOTDT = 5e-5 # seconds
RUNTIME = 1.0 # seconds
injectI = 100e-12 # Amperes

## moose imports
import moose
from moose.neuroml import *
from moose.utils import * # has setupTable(), resetSim() etc
import math

## compartment constants
Em = -65e-3 # V
## if length l and diameter d of cylinder are equal, pi*d*l = 4*pi*r^2 where r=d/2
## i.e. surface area of cylinder is same as surface area of sphere with dia d
length = 100e-6 # 100 microns
diameter = 100e-6 # 100 microns
## specific resistances and capacitance
RM = 100 # Ohm.m^2
CM = 0.01 # F/m^2 # standard value is 1 microF/cm^2
RA = 1 # Ohm.m
## actual resistances and capacitance: derived values
surfaceArea = math.pi*diameter*length # pi from math module
crossSectionArea = math.pi*(diameter/2.0)**2
Rm = RM/surfaceArea
Cm = CM*surfaceArea
Ra = RA*length/crossSectionArea
## firing properties
Vthreshold = Em+5e-3 # Volts # above resting
refractoryPeriod = 2e-3 # seconds
Vreset = Em-2e-3 # Volts #goes below Em after spike

## import numpy and matplotlib in matlab style commands
from pylab import *

def create_LIF():
	## create the integrate and fire neuron
	cells = moose.Neutral('/cells')
	#IF1 = moose.IntFire('/cells/IF1')
	IF1 = moose.LeakyIaF('/cells/IF1')
	IF1.inject = injectI
	IF1.Rm = Rm
	IF1.Cm = Cm
	IF1.Em = Em
	IF1.initVm = Vreset
	IF1.Vthreshold = Vthreshold # firing threshold potential
	IF1.refractoryPeriod = refractoryPeriod # min refractory time before next spike
	IF1.Vreset = Vreset # voltage after spike, typicaly below resting
	## from moose_utils.py sets up a table to record Vm from IF1
	## table is named vmTableIF1 under path of IF1
	return IF1

def run_LIF():
	## reset and run the simulation
	print "Reinit MOOSE."
	## from moose_utils.py sets clocks and resets
	resetSim(['/cells'], SIMDT, PLOTDT)
	print "Running now..."
	moose.start(RUNTIME)

if __name__ == '__main__':
    IF1 = create_LIF()
    IF1vmTable = setupTable("vmTableIF1",IF1,'Vm')

    ## save spikes in table
    table_path = moose.Neutral(IF1.path+'/data').path
    IF1spikesTable = moose.Table(table_path+'/spikesTable')
    IF1spikesTable.threshold = Vthreshold-1e-3
    #moose.connect(IF1,'VmOut',IF1spikesTable,'spike') # This is not edge triggered :(
    moose.connect(IF1,'spike',IF1spikesTable,'input')

    run_LIF()
    print IF1spikesTable.vec
    ## plot the membrane potential of the neuron
    timevec = arange(0.0,RUNTIME+PLOTDT/2.0,PLOTDT)
    figure(facecolor='w')
    plot(timevec, IF1vmTable.vec)
    show()
