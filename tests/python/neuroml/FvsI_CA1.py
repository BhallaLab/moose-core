#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
All SI units
## Plot the firing rate vs current injection curve for a Cerebellar Granule Cell neuron

## Author: Aditya Gilra
## Creation Date: 2012-07-12
## Modification Date: 2012-07-12

                    Wednesday 15 July 2015 09:46:36 AM IST
                    Added unittest
                    Modified for testing with ctest.
"""

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('./../../python/')
import moose
from moose.utils import *
import moose.utils as mu
from moose.neuroml.NeuroML import NeuroML

from pylab import *
import numpy as np

try:
    import unittest2 as unittest
except:
    import unittest

RUNTIME = 1.0 # s

injectmax = 2e-12 # Amperes

neuromlR = NeuroML()
neuromlR.readNeuroMLFromFile('cells_channels/CA1soma.morph.xml')
libcell = moose.Neuron('/library/CA1soma')
CA1Cellid = moose.copy(libcell,moose.Neutral('/cells'),'CA1')
CA1Cell = moose.Neuron(CA1Cellid)
#printCellTree(CA1Cell)

## edge-detect the spikes using spike-gen (table does not have edge detect)
spikeGen = moose.SpikeGen(CA1Cell.path+'/spikeGen')
spikeGen.threshold = -30e-3 # V
CA1CellSoma = moose.Compartment(CA1Cell.path+'/Seg0_soma_0_0')
CA1CellSoma.inject = 0 # by default the cell has a current injection
moose.connect(CA1CellSoma,'VmOut',spikeGen,'Vm')
## save spikes in table
table_path = moose.Neutral(CA1Cell.path+'/data').path
CA1CellSpikesTable = moose.Table(table_path+'/spikesTable')
moose.connect(spikeGen,'spikeOut',CA1CellSpikesTable,'input')

cells_path = '/cells'

## Loop through different current injections
freqList = []

def applyCurrent(currenti):
    global freqList
    moose.reinit()
    CA1CellSoma.inject = currenti
    moose.start(RUNTIME)
    spikesList = array(CA1CellSpikesTable.vector)
    if len(spikesList)>0:
        spikesList = spikesList[where(spikesList>0.0)[0]]
        spikesNow = len(spikesList)
    else: 
        spikesNow = 0.0
    print("For injected current {0}, no of spikes in {1} second: {2}".format(
        currenti, RUNTIME, spikesNow )
        )
    freqList.append( spikesNow/float(RUNTIME) )
    return spikesNow

