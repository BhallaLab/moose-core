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
RUNTIME = 2.0 # seconds
injectI = 1e-8#2.5e-12 # Amperes

## moose imports
import moose
from moose.neuroml import *
from moose.utils import * # has setupTable(), resetSim() etc
import math

## import numpy and matplotlib in matlab style commands
from pylab import *

def create_twoLIFs():
    NML = NetworkML({'temperature':37.0,'model_dir':'.'})
    ## Below returns populationDict = { 'populationname1':(cellname,{instanceid1:moosecell, ... }) , ... }
    ## and projectionDict = { 'projectionname1':(source,target,[(syn_name1,pre_seg_path,post_seg_path),...]) , ... }
    (populationDict,projectionDict) = \
        NML.readNetworkMLFromFile('twoLIFs.net.xml', {}, params={})
    return populationDict,projectionDict

def run_twoLIFs():
	## reset and run the simulation
	print "Reinit MOOSE."
	## from moose_utils.py sets clocks and resets
	resetSim(['/cells[0]'], SIMDT, PLOTDT, simmethod='ee')
	print "Running now..."
	moose.start(RUNTIME)

if __name__ == '__main__':
    populationDict,projectionDict = create_twoLIFs()
    ## element returns the right element and error if not present
    IF1Soma = moose.element(populationDict['LIFs'][1][0].path+'/soma_0')
    IF1Soma.inject = injectI
    IF2Soma = moose.element(populationDict['LIFs'][1][1].path+'/soma_0')
    IF2Soma.inject = 0.0#injectI*2.0
    IF2Soma.inject = injectI
    IF1vmTable = setupTable("vmTableIF1",IF1Soma,'Vm')
    IF2vmTable = setupTable("vmTableIF2",IF2Soma,'Vm')

    ## edge-detect the spikes using spike-gen (table does not have edge detect)
    ## IaF_spikegen is already present for compartments having IaF mechanisms
    spikeGen = moose.SpikeGen(IF1Soma.path+'/IaF_spikegen')
    table_path = moose.Neutral(IF1Soma.path+'/data').path
    IF1spikesTable = moose.Table(table_path+'/spikesTable')
    moose.connect(spikeGen,'spikeOut',IF1spikesTable,'input') ## spikeGen gives spiketimes

    run_twoLIFs()
    print "Spiketimes :",IF1spikesTable.vector
    ## plot the membrane potential of the neuron
    timevec = arange(0.0,RUNTIME+PLOTDT/2.0,PLOTDT)
    figure(facecolor='w')
    print IF1vmTable,IF2vmTable
    plot(timevec, IF1vmTable.vector,'r-')
    figure(facecolor='w')
    plot(timevec, IF2vmTable.vector,'b-')
    show()

    ## At the end, some issue with Func (as per Subha) gives below or core dump error
    ## *** glibc detected *** python: corrupted double-linked list: 0x00000000038f9aa0 ***
