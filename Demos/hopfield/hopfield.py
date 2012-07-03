#PARTIAL DEMO
# demo of a simple hopfield network using leaky integrate and fire neurons
# memory.csv has the memory being saved, the synaptic weights are set at start according to this memory
# memory must be a square matrix of 0's and 1's only
# input.csv has partial input given as input to the cells.
# - By C.H.Chaitanya
# This code is available under GPL3 or greater GNU license 

import moose
import math
from moose.utils import *
import csv
import numpy as np
#import matplotlib.pyplot as plt
from pylab import *

def readMemory(memoryFile):
    f = open(memoryFile,'r')
    testLine = f.readline()
    dialect = csv.Sniffer().sniff(testLine) #to get the format of the csv
    f.close()
    f = open(memoryFile, 'r')
    reader = csv.reader(f,dialect)
    memory = []
    for row in reader:
        for i in row[0:]:
            memory.append(int(i))
    f.close()
    return memory

def updateWeights(memory,weightMatrix):
    for i in range(len(memory)):
        newWeights = []
        for j in range(len(memory)):
            if i != j:
                newWeights.append(memory[i]*memory[j])
            else:
                newWeights.append(0)
        #add the new synaptic weights to the old ones.
        weightMatrix[i*len(memory):(i+1)*len(memory)] = [sum(a) for a in zip(*([weightMatrix[i*len(memory):(i+1)*len(memory)]]+[newWeights]))] 
    return weightMatrix

def createNetwork(synWeights,inputGiven):
    numberOfCells = int(math.sqrt(len(synWeights)))
    cells = []
    Vms   = []
    inTables = []
    cellNameObjectDict = {}
    hopfield = moose.Neutral('/hopfield')
    pg = moose.PulseGen('/hopfield/inPulGen')

    pgTable = moose.Table('/hopfield/inPulGen/pgTable')
    moose.connect(pgTable, 'requestData', pg, 'get_output')
    pg.count = 1
    pg.level[0] = 3
    pg.width[0] = 2e-03
    pg.delay[0] = 5e-02 #every 50ms

    for i in range(numberOfCells):
        cellPath = '/hopfield/cell_'+str(i+1)
        cell = moose.IntFire(cellPath) #non programmer friendly numbering
        cell.setField('tau',10e-3)
        cell.setField('Vm', -0.07)
        #cell.setField('refractoryPeriod', 0.1)
        #cell.setField('thresh', 0.0)
        cell.synapse.num = numberOfCells+1 
        #number of synapses - for user friendly numbering - ignore synapse[0]
        cell.synapse[i+1].weight = 4
        cell.synapse[i+1].delay = 1e-3

        VmVals = moose.Table(cellPath+'/Vm_cell_'+str(i+1))
        moose.connect(VmVals, 'requestData', cell, 'get_Vm')

        inSpkGen = moose.SpikeGen(cellPath+'/inSpkGen')
        inSpkGen.setField('threshold', 2.0)
        inSpkGen.setField('edgeTriggered', True)

        if inputGiven[i] == 1:
            moose.connect(pg, 'outputOut', inSpkGen, 'Vm')

            inTable = moose.Table(cellPath+'/inSpkGen/inTable')
            moose.connect(inTable, 'requestData', inSpkGen, 'get_hasFired')
            inTables.append(inTable)

        moose.connect(inSpkGen, 'event', cell.synapse[i+1] ,'addSpike') #self connection is the input 
        cellNameObjectDict[cellPath] = cell
        Vms.append(VmVals)
        cells.append(cell)

    for currentCell in range(numberOfCells):
        for connectCell in range(numberOfCells):
            if currentCell != connectCell: #no self connections
                connSyn = cellNameObjectDict['/hopfield/cell_'+str(connectCell+1)].synapse[connectCell+1]
                connSyn.weight = synWeights[currentCell*numberOfCells + connectCell]
                connSyn.delay = 20e-3
                moose.connect(cellNameObjectDict['/hopfield/cell_'+str(currentCell+1)], 'spike', connSyn, 'addSpike')

    return cells,Vms,pgTable,inTables

#def saveMemories([memoryFiles]):
#    synWeights = [0]*len(memories[0])*len(memories[0])
#    for memory in memories:
#        read
    

memoryFile1 = "memory1.csv"
memory = readMemory(memoryFile1)
synWeights = updateWeights(memory,[0]*len(memory)*len(memory))

memoryFile2 = "memory2.csv"
memory2 = readMemory(memoryFile2)
synWeights = updateWeights(memory2,synWeights)

inputFile = "input.csv"
cells,Vms,pgTable,inTables = createNetwork(synWeights,readMemory(inputFile))

# moose.setClock(0, 1e-4)
# moose.useClock(0, '/hopfield/inPulGen/pgTable,/hopfield/inPulGen,','process')
# moose.useClock(0, '/hopfield/##[TYPE=IntFire],/hopfield/##[TYPE=Table],/hopfield/##[TYPE=SpikeGen]', 'process')
# moose.reinit()
# moose.start(0.2)

# #plot(pgTable.vec[1:])
# #for yset,inTable in enumerate(inTables):
# #    plot(float(yset)+inTable.vec[1:])
# for ySet,Vm in enumerate(Vms):
#     plot(float(2*ySet)/(1e+7)+Vm.vec[1:])
# #plot(Vms[0].vec[1:])
# show()
