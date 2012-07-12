# -*- coding: utf-8 -*-
## all SI units
########################################################################################
## Plot the firing rate vs current injection curve for a Cerebellar Granule Cell neuron
## Author: Aditya Gilra
## Creation Date: 2012-07-12
## Modification Date: 2012-07-12
########################################################################################

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')
import moose
from moose.utils import *

from moose.neuroml.NeuroML import NeuroML

from pylab import *

SIMDT = 25e-6 # s
PLOTDT = 25e-6 # s
RUNTIME = 2.0 # s

injectmax = 20e-12 # Amperes

neuromlR = NeuroML()
neuromlR.readNeuroMLFromFile('cells_channels/Granule_98.morph.xml')
libcell = moose.Neuron('/library/Granule_98')
granCellid = moose.copy(libcell,moose.Neutral('/cells'),'granCell')
granCell = moose.Neuron(granCellid)

## edge-detect the spikes using spike-gen (table does not have edge detect)
spikeGen = moose.SpikeGen(granCell.path+'/spikeGen')
spikeGen.threshold = 0e-3 # V
granCellSoma = moose.Compartment(granCell.path+'/Soma_0')
moose.connect(granCellSoma,'VmOut',spikeGen,'Vm')
## save spikes in table
table_path = moose.Neutral(granCell.path+'/data').path
granCellSpikesTable = moose.Table(table_path+'/spikesTable')
moose.connect(spikeGen,'event',granCellSpikesTable,'input')

## from moose_utils.py sets clocks and resets/reinits
resetSim(['/cells'], SIMDT, PLOTDT)

## Loop through different current injections
freqList = []
currentvec = arange(0.0, injectmax, injectmax/100.0)
for currenti in currentvec:
    moose.reinit()
    granCellSoma.inject = currenti
    moose.start(RUNTIME)
    spikesList = array(granCellSpikesTable.vec)
    if len(spikesList)>0:
        spikesList = spikesList[where(spikesList>0.0)[0]]
        spikesNow = len(spikesList)
    else: spikesNow = 0.0
    print "For injected current =",currenti,\
        "number of spikes in",RUNTIME,"seconds =",spikesNow
    freqList.append( spikesNow/float(RUNTIME) )

## plot the F vs I curve of the neuron
figure(facecolor='w')
plot(currentvec, freqList,'o-')
xlabel('time (s)')
ylabel('frequency (Hz)')
title('HH single-compartment Cell')
show()
