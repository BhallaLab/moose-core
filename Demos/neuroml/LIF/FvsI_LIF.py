#!/usr/bin/python
# -*- coding: utf-8 -*-
## all SI units
########################################################################################
## Plot the firing rate vs current injection curve for a leaky integrate and fire neuron
## Author: Aditya Gilra
## Creation Date: 2012-06-08
## Modification Date: 2012-06-08
########################################################################################

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('.')

from LIFxml_firing import *
injectmax = 10e-12 # Amperes

IF1 = create_LIF()

## edge-detect the spikes using spike-gen (table does not have edge detect)
spikeGen = moose.SpikeGen(IF1.path+'/spikeGen')
spikeGen.threshold = IF1.Vthreshold
moose.connect(IF1,'VmOut',spikeGen,'Vm')
## save spikes in table
table_path = moose.Neutral(IF1.path+'/data').path
IF1spikesTable = moose.Table(table_path+'/spikesTable')
moose.connect(spikeGen,'event',IF1spikesTable,'input')

## from moose_utils.py sets clocks and resets/reinits
resetSim(['/cells'], SIMDT, PLOTDT, simmethod='hsolve')

## Loop through different current injections
freqList = []
currentvec = arange(0.4e-12, injectmax, injectmax/30.0)
### log scale for x-axisx
#dlogI = log(2.5)
#logcurrentvec = arange(log(injectmax)-30*dlogI,log(injectmax),dlogI)
#currentvec = [0.0]
#currentvec.extend( [exp(I) for I in logcurrentvec] )
for currenti in currentvec:
    moose.reinit()
    IF1.inject = currenti
    moose.start(RUNTIME)
    spikesList = array(IF1spikesTable.vec)
    if len(spikesList)>0:
        spikesList = spikesList[where(spikesList>0.0)[0]]
        spikesNow = len(spikesList)
    else: spikesNow = 0.0
    print "For injected current =",currenti,\
        "number of spikes in",RUNTIME,"seconds =",spikesNow
    freqList.append( spikesNow/float(RUNTIME) )

## plot the F vs I curve of the neuron
figure(facecolor='w')
plot(currentvec, freqList,'o-',linewidth=2)
xlabel('current (A)',fontsize=24)
ylabel('frequency (Hz)',fontsize=24)
title('Leaky Integrate and Fire',fontsize=24)
show()
