# -*- coding: utf-8 -*-
## all SI units
########################################################################################
## Plot the firing rate vs current injection curve for a leaky integrate and fire neuron
## Author: Aditya Gilra
## Creation Date: 2012-06-08
## Modification Date: 2012-06-08
########################################################################################

from LIF_firing import *
injectmax = 1000e-12 # Amperes

IF1 = create_LIF()

## save spikes in table
table_path = moose.Neutral(IF1.path+'/data').path
IF1spikesTable = moose.Table(table_path+'/spikesTable')
IF1spikesTable.threshold = Vthreshold-1e-3
moose.connect(IF1,'spike',IF1spikesTable,'input')

## from moose_utils.py sets clocks and resets/reinits
resetSim(['/cells'], SIMDT, PLOTDT)

## Loop through different current injections
freqList = []
currentvec = arange(0.0, injectmax, injectmax/30.0)
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
plot(currentvec, freqList,'o-')
xlabel('time (s)')
ylabel('frequency (Hz)')
title('Leaky Integrate and Fire')
show()
