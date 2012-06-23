#PARTIAL DEMO
# demo of a simple hopfield network using leaky integrate and fire neurons
# memory.csv has the memory being saved, the synaptic weights are set at start according to this memory
# memory must be a square matrix of 0's and 1's only
# input.csv has partial input given as input to the cells.
# - By C.H.Chaitanya
# This code is available under GPL3 or greater GNU license 

import moose
import csv
import numpy as np

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
        tempList = []
        for j in range(len(memory)):
            if i != j:
                tempList.append(memory[i]*memory[j])
            else:
                tempList.append(0)
        weightMatrix.append(tempList)
    return weightMatrix

def createNetwork(synWeights,inputGiven):
    numberOfCells = len(synWeights[0])
    cells = []

    pg = moose.PulseGen('/inPulGen')
    pg.count = 1
    pg.level[0] = 2.0
    pg.width[0] = 0.1
    pg.delay[0] = 5e-02 #every 50ms

    for i in range(numberOfCells):
        cell = moose.IntFire('/cell_'+str(i+1)) #non programmer friendly numbering
        cell.setField('tau',10.0)
        cell.setField('Vm', -0.07)
        #cell.setField('refractoryPeriod', 0.1)
        #cell.setField('thresh', 0.0)
        cell.synapse.num = numberOfCells+1 
        #number of synapses - for user friendly numbering - ignore synapse[0]
        cell.synapse[i+1].weight = 1
        cell.synapse[i+1].delay = 1e-3

        inSpkGen = moose.SpikeGen('/cell_'+str(i+1)+'/inSpkGen')
        moose.connect(inSpkGen, 'event', cell.synapse[i+1] ,'addSpike') #self connection is the input 

        if inputGiven[i] == 1:
            moose.connect(pg, 'outputOut', moose.element('/cell_'+str(i+1)+'/inSpkGen'), 'Vm')

        cells.append(cell)

    for currentCell in range(numberOfCells): #currCell 
        for connectCell in range(numberOfCells): #connectToCell
            if currentCell != connectCell: #no self connections
                connSyn = moose.element('/cell_'+str(connectCell+1)).synapse[connectCell+1]
                connSyn.weight = synWeights[currentCell][connectCell]
                connSyn.delay = 2e-3
                moose.connect(moose.element('/cell_'+str(currentCell+1)), 'spike', connSyn, 'addSpike')

    return cells

memoryFile1 = "memory1.csv"
memory = readMemory(memoryFile1)
synWeights = updateWeights(memory,[])

memoryFile2 = "memory2.csv"
memory2 = readMemory(memoryFile2)
synWeights = updateWeights(memory2,synWeights)

inputFile = "input.csv"
cells = createNetwork(synWeights,readMemory(inputFile))

