# -*- coding: utf-8 -*-
## all SI units
########################################################################################
## Plot the firing rate vs current injection curve for a leaky integrate and fire neuron
## Author: Aditya Gilra
## Creation Date: 2012-06-08
## Modification Date: 2012-06-08
########################################################################################

## simulation parameters
SIMDT = 5e-5 # seconds
PLOTDT = 5e-5 # seconds
RUNTIME = 1.0 # seconds
injectI = 200e-12 # Amperes

## moose imports
import moose
mcontext = moose.context
from moose.neuroml import *
from moose.utils import * # has setupTable(), resetSim() etc
import math

## compartment constants
Em = -65e-3 # V
## if length l and diameter d of cylinder are equal, pi*d*l = 4*pi*r^2 where r=d/2
## i.e. surface area of cylinder is same as surface area of sphere with dia d
length = 50e-6 # 100 microns
diameter = 50e-6 # 100 microns
## specific resistances and capacitance
RM = 1 # Ohm.m^2
CM = 0.01 # F/m^2 # standard value is 1 microF/cm^2
RA = 1 # Ohm.m
## actual resistances and capacitance: derived values
surfaceArea = math.pi*diameter*length # pi from math module
crossSectionArea = math.pi*(diameter/2.0)**2
Rm = RM/surfaceArea
Cm = CM*surfaceArea
Ra = RA*length/crossSectionArea
## firing properties
Vthreshold = Em + 10e-3 # Volts # above resting
refractT = 5e-3 # seconds
refractV = Em-5e-3 # Volts # goes below Em after spike

## import numpy and matplotlib in matlab style commands
from pylab import *

## create the integrate and fire neuron
IF1 = moose.IntFire('/cells/IF1')
IF1.Rm = Rm
IF1.Cm = Cm
IF1.Em = Em
IF1.initVm = Em
IF1.Vt = Vthreshold # firing threshold potential
IF1.refractT = refractT # min refractory time before next spike
IF1.Vr = refractV # voltage after spike, typicaly below resting

## from moose_utils.py sets up a table to record Vm from IF1
## table is named vmTableIF1 under path of IF1
IF1.vmTable = setupTable("vmTableIF1",IF1,'Vm')
IF1.vmTable.stepMode = TAB_SPIKE
IF1.vmTable.stepSize = Vthreshold-1e-3


## reset and run the simulation
print "Resetting MOOSE."
## from moose_utils.py sets clocks and resets
resetSim(mcontext, SIMDT, PLOTDT)

## Loop through different current injections
spikesTillNow = 0
freqList = []
currentvec = arange(0.0, injectI, injectI/30.0)
for currenti in currentvec:
    IF1.inject = currenti
    print "Running for injected current =",currenti
    mcontext.step(RUNTIME)
    spikesList = array(IF1.vmTable)
    spikesList = spikesList[where(spikesList>0.0)[0]]
    spikesNow = len(spikesList) - spikesTillNow
    freqList.append( spikesNow/float(RUNTIME) )
    spikesTillNow = len(spikesList)

## plot the F vs I curve of the neuron
figure(facecolor='w')
plot(currentvec, freqList)
show()
